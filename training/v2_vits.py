#!/usr/bin/env python3
"""Private single-speaker VITS adaptation, reconstruction and file synthesis.

Uses the BSD-3 pretrained CV Portuguese generator with its exact grapheme IDs;
Coqui 0.28 compatibility runtime, native 22050Hz HiFi-GAN decoder, and upstream
GAN/CFM losses. No service control, downloading or CPU inference offloading.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/var/tmp/aegis-v2-work/vits')
DATA = Path('/var/tmp/aegisvoice-v2-training-20261102/prepared')
BASE_SHA = '2002176aa93a244b63b70618ba6ddc45d7c649661c99167d143e7d7d96f0c258'
TEXT_POLICY = 'vits-f5-refinement-1-reviewed-text-punctuation-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(root, kind, **values):
    row = dict(event=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    encoded=json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'
    def append_bounded(path):
        with path.open('a') as stream:stream.write(encoded)
        if path.stat().st_size > 32*1024**2:
            with path.open('r+b') as stream:
                stream.seek(-4*1024**2,2);tail=stream.read().split(b'\n',1)[-1]
                stream.seek(0);stream.write(tail);stream.truncate()
    append_bounded(root/'events.jsonl')
    if kind == 'update':
        atomic_json(root/'progress.json',row)
        compact=values['step']<=4 or values['step']%10==0
        if compact:append_bounded(root/'steps.jsonl')
    else:
        append_bounded(root/'milestones.jsonl');compact=True
    if compact:
        append_bounded(root/'training.log')
        print(encoded.rstrip(),flush=True)


def bootstrap(args):
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.device == 'cuda':
        if not args.gpu_authorized:
            raise ValueError('--gpu-authorized required; use the external isolated GPU worker')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if apps and any(int(p) != os.getpid() for p in apps.splitlines()):
            raise RuntimeError('another compute process occupies GPU')
    sys.path.insert(0, str(args.root / 'deps'))
    sys.path.insert(0, str(REPO / 'gateway'))
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    from TTS.config import load_config
    from TTS.tts.models.vits import Vits
    base = args.root / 'model/tts_models--pt--cv--vits'
    if digest(base / 'model_file.pth.tar') != BASE_SHA:
        raise ValueError('unrecognized pretrained generator')
    config = load_config(base / 'config.json')
    config.model_args.speakers_file = config.speakers_file = str(base / 'speaker_ids.json')
    config.model_args.language_ids_file = config.language_ids_file = str(base / 'language_ids.json')
    config.model_args.init_discriminator = False
    model = Vits(config)
    # The official file is a trusted published checkpoint with config metadata.
    base_checkpoint = torch.load(base / 'model_file.pth.tar', map_location='cpu', weights_only=False)
    model.load_state_dict(base_checkpoint['model'], strict=True)
    del base_checkpoint
    # Experimental normalization is separate from the frozen Qwen profile.
    # The owner target is A-é-jis, while the protected legacy lexicon uses Égide.
    from training.f5_refine_text import normalize_text
    from types import SimpleNamespace
    normalizer = SimpleNamespace(normalize=lambda text: SimpleNamespace(normalized_text=normalize_text(text)['tts_text']))
    return torch, np, model, config, normalizer


def tokens(model, normalizer, text):
    text = normalizer.normalize(text).normalized_text
    model.tokenizer.not_found_characters.clear()
    ids = model.tokenizer.text_to_ids(text, language='pt')
    if model.tokenizer.not_found_characters or not ids:
        # No private transcript is printed by this runner.
        raise ValueError('grapheme frontend has OOV characters: ' + repr(model.tokenizer.not_found_characters))
    return ids


def rows_and_fingerprint(prepared, model, normalizer):
    rows = {}
    hashes = {'normalizer_code_sha256': digest(REPO / 'training/f5_refine_text.py'),'text_policy':TEXT_POLICY}
    for split in ('train', 'validation', 'test'):
        path = prepared / (split + '.jsonl')
        hashes[split] = digest(path)
        rows[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_partitions(rows)
    selection=prepared/'selection-audit.json'
    if selection.exists():
        declared=json.loads(selection.read_text())['selected_manifest_hashes']
        if any(declared[s] != hashes[s] for s in ('train','validation','test')):
            raise ValueError('frozen VITS selection manifest hash mismatch')
        hashes['selection_audit_sha256']=digest(selection)
    for split in ('train','validation'):
        for row in rows[split]:
            row['_tokens'] = tokens(model, normalizer, row.get('text',row['original_text']))
    return rows, hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def validate_partitions(rows):
    """Reject duplicate-ID appended corrections before waveform/token loading."""
    import re
    import unicodedata
    ids=set();seen={kind:{} for kind in ('group','text','audio')}
    for split in ('train','validation','test'):
        if not rows.get(split):raise ValueError('empty partition: '+split)
        for row in rows[split]:
            identifier=row['id']
            if identifier in ids:raise ValueError('duplicate ID within/across partitions: '+identifier)
            ids.add(identifier)
            text=row.get('text',row.get('original_text',''))
            if not text.strip():raise ValueError('empty transcript: '+identifier)
            canonical=' '.join(re.findall(r'\w+',unicodedata.normalize('NFC',text).casefold()))
            keys=dict(group=row.get('group_id',identifier),text=canonical,
                      audio=row.get('source_audio_sha256',row.get('sha256')))
            if not keys['audio']:raise ValueError('missing source audio hash: '+identifier)
            for kind,key in keys.items():
                previous=seen[kind].setdefault(key,split)
                if previous != split:raise ValueError('cross-split '+kind+' leakage: '+identifier)


def get_audio(row, np):
    import soundfile as sf
    from scipy.signal import resample_poly
    # Target data: original treated 48kHz. Public language stage: its existing
    # hash-bound 24kHz prepared file, explicitly identified as a derivative.
    if row.get('source_audio_filepath'):
        path, expected = row['source_audio_filepath'], row.get('source_audio_sha256')
    else:
        path, expected = row['audio_filepath'], row.get('sha256')
    if not expected or digest(path) != expected:
        raise ValueError('input audio SHA256 mismatch or missing binding: ' + row['id'])
    audio, rate = sf.read(path, dtype='float32')
    declared_rate = row.get('prepared_rate') if not row.get('source_audio_filepath') else row.get('source_rate')
    if declared_rate is not None and rate != declared_rate:
        raise ValueError('input audio sample-rate metadata mismatch: ' + row['id'])
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('nonfinite/stereo input')
    if rate != 22050:
        divisor = math.gcd(rate, 22050)
        audio = resample_poly(audio, 22050 // divisor, rate // divisor, window=('kaiser', 10)).astype(np.float32)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError('resampling exceeds valid range; no clipping applied')
    return audio


def batch(row, model, torch, np, device):
    audio = get_audio(row, np)
    token = torch.tensor(row['_tokens'], dtype=torch.long, device=device)[None]
    waveform = torch.from_numpy(audio.copy()).to(device)[None, None]
    value = dict(tokens=token, token_lens=torch.tensor([token.shape[1]], device=device),
                 waveform=waveform, waveform_rel_lens=torch.ones(1, device=device),
                 speaker_ids=torch.zeros(1, dtype=torch.long, device=device),
                 language_ids=torch.zeros(1, dtype=torch.long, device=device), d_vectors=None)
    return model.format_batch_on_device(value)


def save_audio(path, values, np, rate=22050):
    import soundfile as sf
    from scipy.signal import resample_poly
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if rate != 24000:
        divisor = math.gcd(rate, 24000)
        values = resample_poly(values, 24000 // divisor, rate // divisor, window=('kaiser', 10)).astype(np.float32)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('nonfinite/empty synthesis')
    sf.write(path, values, 24000, subtype='FLOAT')
    return dict(path=str(path), sha256=digest(path), sample_rate=24000, seconds=len(values) / 24000,
                peak=float(np.abs(values).max()), outside_unit=int((np.abs(values) > 1).sum()))


def load_trained(args, model, torch):
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        weights = checkpoint['model']
        generator = {key: value for key, value in weights.items() if not key.startswith('disc.')}
        model.load_state_dict(generator, strict=True)
        return checkpoint.get('step', 0)
    return 0


def infer(args, torch, np, model, normalizer):
    step = load_trained(args, model, torch)
    model.to(args.device).eval()
    if args.cases:
        cases = json.loads(args.cases.read_text())
    else:
        cases = [dict(id='novel', text=args.text)]
    outputs = []
    for i, case in enumerate(cases):
        # Cache neither frontend nor audio. Native file mode exposes one complete
        # waveform; count final-rate resampling in the measured synthesis cost.
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        token = torch.tensor(tokens(model, normalizer, case['text']), device=args.device)[None]
        output = model.inference(token, aux_input=dict(speaker_ids=torch.zeros(1, dtype=torch.long, device=args.device),
                                                      language_ids=torch.zeros(1, dtype=torch.long, device=args.device)))
        audio = output['model_outputs'].squeeze().float().cpu().numpy()
        # Save includes conversion and I/O, so this is conservative file ready.
        artifact = save_audio(args.run / (case['id'] + '.wav'), audio, np)
        elapsed = time.perf_counter() - start
        outputs.append(dict(id=case['id'], cold_generation=i == 0, step=step,
                            seconds=elapsed, ttfa_scope='complete file available; no HTTP or streaming claim',
                            first_pcm_ms=elapsed * 1000, rtf=elapsed / artifact['seconds'], audio=artifact,
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None))
    atomic_json(args.run / 'inference.json', outputs)
    event(args.run, 'inference_complete', count=len(outputs), step=step)


def reconstruct(args, torch, np, model, normalizer):
    rows, fingerprint = rows_and_fingerprint(args.prepared, model, normalizer)
    model.to(args.device).eval()
    # Three deterministic validation originals; no optimization on final test.
    pool = sorted(rows['validation'], key=lambda r: (r['duration'], r['id']))
    selected = [pool[i * (len(pool) - 1) // 2] for i in range(3)]
    result = []
    for row in selected:
        torch.manual_seed(args.seed)
        b = batch(row, model, torch, np, args.device)
        start = time.perf_counter()
        with torch.inference_mode():
            g = model.emb_g(b['speaker_ids']).unsqueeze(-1)
            z, _, _, mask = model.posterior_encoder(b['spec'], b['spec_lens'], g=g)
            audio = model.waveform_decoder(z * mask, g=g).squeeze().float().cpu().numpy()
        reconstructed = save_audio(args.run / (row['id'] + '-posterior.wav'), audio, np)
        original = save_audio(args.run / (row['id'] + '-input.wav'), get_audio(row, np), np)
        result.append(dict(id=row['id'], seconds=time.perf_counter() - start, original=original,
                           reconstructed=reconstructed, scope='VITS posterior latent -> integrated HiFi-GAN decoder; not an external mel vocoder roundtrip'))
    atomic_json(args.run / 'reconstruction.json', dict(dataset=fingerprint, results=result))
    event(args.run, 'reconstruction_complete', count=len(result))


def rng_state(torch, np):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value, torch, np):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value['cuda'])


def train(args, torch, np, model, config, normalizer):
    if args.device != 'cuda' and not args.allow_cpu_train:
        raise ValueError('training defaults to CUDA; CPU debug needs --allow-cpu-train')
    from TTS.tts.layers.vits.discriminator import VitsDiscriminator
    from TTS.tts.models.vits import wav_to_mel
    from TTS.tts.utils.helpers import segment
    rows, fingerprint = rows_and_fingerprint(args.prepared, model, normalizer)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    initialization = None
    if args.init_checkpoint:
        source = torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict({k:v for k,v in source['model'].items() if not k.startswith('disc.')},strict=True)
        initialization = dict(path=str(args.init_checkpoint.resolve()),sha256=digest(args.init_checkpoint),
                              source_step=source['step'],policy='strict generator-only initialization; fresh critic/optimizers/scheduler/RNG/data order for new stage')
        del source
    # The published checkpoint contains the complete generator but no critic.
    # Add a new standard multi-period critic only after strict generator load.
    model.disc = VitsDiscriminator(periods=config.model_args.periods_multi_period_discriminator,
                                   use_spectral_norm=config.model_args.use_spectral_norm_disriminator)
    model.args.init_discriminator = True
    model.to(args.device)
    gen_parameters = [p for name, p in model.named_parameters() if not name.startswith('disc.')]
    disc_parameters = list(model.disc.parameters())
    optimizers = [torch.optim.AdamW(disc_parameters, lr=args.lr_disc, betas=(.8, .99), eps=1e-9),
                  torch.optim.AdamW(gen_parameters, lr=args.lr_gen, betas=(.8, .99), eps=1e-9)]
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=.999875) for o in optimizers]
    criteria = [c.to(args.device) for c in model.get_criterion()]
    # FP32 is the initial qualified route. BF16 is optional and measured alone.
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    step, epoch, cursor, best, stale, reduced = 0, 0, 0, math.inf, 0, False
    permutation = []
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['dataset_fingerprint'] != fingerprint or state['precision'] != args.precision:
            raise ValueError('resume dataset/precision mismatch')
        if state.get('stage','target') != args.stage or state.get('initialization') != initialization:
            raise ValueError('resume stage/initialization mismatch')
        model.load_state_dict(state['model'], strict=True)
        for optimizer, saved in zip(optimizers, state['optimizers']): optimizer.load_state_dict(saved)
        for scheduler, saved in zip(schedulers, state['schedulers']): scheduler.load_state_dict(saved)
        scaler.load_state_dict(state['scaler'])
        step, epoch, cursor = state['step'], state['epoch'], state['cursor']
        permutation, best, stale, reduced = state['permutation'], state['best'], state['stale'], state['reduced']
        restore_rng(state['rng'], torch, np)
        event(args.run, 'resumed', step=step, epoch=epoch, cursor=cursor, checkpoint_sha256=digest(args.resume))
    atomic_json(args.run / 'recipe.json', dict(base_sha256=BASE_SHA, dataset_fingerprint=fingerprint,
                                              stage=args.stage, initialization=initialization,
                                              train_count=len(rows['train']), validation_count=len(rows['validation']),
                                              precision=args.precision, lr_gen=args.lr_gen, lr_disc=args.lr_disc,
                                              scheduler_cadence='ExponentialLR gamma .999875 once per completed epoch',
                                              seed=args.seed, sample_rate=22050, final_sample_rate=24000,
                                              code_sha256=digest(__file__), tokenizer='checkpoint graphemes + f5-refinement-1 experimental PT-BR normalizer; A-é-jis owner target',
                                              checkpoint_policy='atomic last and best, cadence <= checkpoint_seconds plus epoch end',
                                              selection='fixed validation reconstruction mel L1; synthetic validation separately; patience8 relative.5%, one LR reduction.3',
                                              input_audio_policy='prefer original source_audio_filepath with source_audio_sha256; otherwise public prepared audio_filepath with sha256 and prepared_rate; all resampled directly from chosen input to22050',
                                              text_probe_policy='all7 at initialization; language stage all7 everyepoch keeping6latest epoch dirs plus initial/best/last; target command/acronyms/long at everybest and epoch3/6/etc plus pilot boundaries',
                                              licenses=dict(code='MPL-2.0', pretrained_weights='BSD-3-Clause')))
    stop = False
    def on_signal(sig, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, on_signal)
    last_save = time.monotonic()
    def save(name):
        nonlocal last_save
        # Peak overlap ~two optimizer checkpoints. Retain old until atomic swap.
        free = os.statvfs(args.run).f_bavail * os.statvfs(args.run).f_frsize
        if free < 6 * 1024**3:
            raise RuntimeError('insufficient disk margin for atomic checkpoint')
        target = args.run / name; temporary = target.with_suffix('.pt.tmp')
        state = dict(model=model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
                     schedulers=[s.state_dict() for s in schedulers], scaler=scaler.state_dict(),
                     rng=rng_state(torch, np), step=step, epoch=epoch, cursor=cursor,
                     permutation=permutation, best=best, stale=stale, reduced=reduced,
                     dataset_fingerprint=fingerprint, precision=args.precision,
                     stage=args.stage,initialization=initialization,
                     config=config.to_dict(), code_sha256=digest(__file__))
        with temporary.open('wb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
        atomic_json(args.run / (name + '.json'), dict(step=step, epoch=epoch, cursor=cursor,
                                                    bytes=target.stat().st_size, sha256=digest(target)))
        last_save = time.monotonic()
        event(args.run, 'checkpoint', path=str(target), step=step, epoch=epoch)
    def validation():
        saved_rng = rng_state(torch, np); values = []
        model.eval()
        with torch.no_grad():
            for i, row in enumerate(rows['validation']):
                torch.manual_seed(args.seed + 100000 + i)
                b = batch(row, model, torch, np, args.device)
                output = model.forward(b['tokens'], b['token_lens'], b['spec'], b['spec_lens'], b['waveform'],
                                       aux_input={k:b[k] for k in ('speaker_ids','language_ids','d_vectors')})
                target = segment(b['mel'], output['slice_ids'], model.spec_segment_size, pad_short=True)
                mel = wav_to_mel(output['model_outputs'].float(), config.audio.fft_size,
                                 config.audio.sample_rate, config.audio.num_mels, config.audio.hop_length,
                                 config.audio.win_length, config.audio.mel_fmin, config.audio.mel_fmax, center=False)
                loss = torch.nn.functional.l1_loss(mel, target).item()
                if not math.isfinite(loss): raise FloatingPointError('nonfinite validation')
                values.append(loss)
        restore_rng(saved_rng, torch, np); model.train()
        score = sum(values) / len(values)
        event(args.run, 'validation', step=step, epoch=epoch, mel_l1=score, count=len(values))
        return score
    def text_probes(label, all_cases=False):
        # Synthetic checks are distinct from posterior reconstruction loss.
        # Bounded retention: initial seven; latest best and latest three only.
        saved_rng = rng_state(torch, np); model.eval()
        cases = json.loads((REPO/'benchmarks/v2/validation.pt-BR.json').read_text())
        if not all_cases:
            cases = [c for c in cases if c['id'] in ('v_command','v_acronyms','v_long')]
        destination = args.run/'probes'/label
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        results = []
        for case in cases:
            torch.manual_seed(40733133)
            started = time.perf_counter()
            ids = torch.tensor(tokens(model, normalizer, case['text']), device=args.device)[None]
            output = model.inference(ids, aux_input=dict(speaker_ids=torch.zeros(1,dtype=torch.long,device=args.device),
                                                        language_ids=torch.zeros(1,dtype=torch.long,device=args.device)))
            values = output['model_outputs'].squeeze().float().cpu().numpy()
            # Atomic replacement prevents an interrupted probe from destroying
            # the preceding complete audio for the same comparison slot.
            target = destination/(case['id']+'.wav')
            temporary = destination/(case['id']+'.temporary.wav')
            artifact = save_audio(temporary, values, np)
            os.replace(temporary,target);artifact['path']=str(target)
            results.append(dict(case=case['id'],step=step,epoch=epoch,seconds=time.perf_counter()-started,
                                effective_text=normalizer.normalize(case['text']).normalized_text,
                                raw_native_peak=float(np.abs(values).max()),native_samples=len(values),
                                audio=artifact,identity='not assessed by this numeric probe'))
        atomic_json(destination/'results.json',dict(step=step,epoch=epoch,results=results,
                                                  metric_scope='text->waveform, complete file; no HTTP; raw native peak and final FLOAT audio',
                                                  checkpoint='best.pt' if label=='best' else 'last.pt' if label=='last' or label.startswith('epoch-') else 'initial generator'))
        restore_rng(saved_rng,torch,np);model.train()
        event(args.run,'text_probes',step=step,epoch=epoch,label=label,count=len(results),
              output=str(destination/'results.json'),outside_unit=sum(r['audio']['outside_unit'] for r in results))
    if not args.resume:
        best = validation(); save('best.pt'); save('last.pt'); text_probes('initial',all_cases=True)
    while not stop:
        if cursor >= len(permutation):
            epoch += 1; cursor = 0
            permutation = list(range(len(rows['train']))); random.shuffle(permutation)
        model.train()
        while cursor < len(permutation) and not stop:
            row = rows['train'][permutation[cursor]]
            started = time.perf_counter()
            b = batch(row, model, torch, np, args.device)
            probe = next(model.text_encoder.parameters()); before = probe.detach().clone()
            for o in optimizers: o.zero_grad(set_to_none=True)
            with torch.autocast(args.device, dtype=dtype, enabled=args.precision == 'bf16'):
                _, disc_losses = model.train_step(b, criteria[0], optimizer_idx=0)
            if not torch.isfinite(disc_losses['loss']): raise FloatingPointError('nonfinite discriminator')
            disc_losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(disc_parameters, 1000., error_if_nonfinite=True)
            optimizers[0].step()
            # Freeze critic parameters for the generator gradient, retaining the
            # waveform path and feature matching derivative through the critic.
            for p in disc_parameters: p.requires_grad_(False)
            with torch.autocast(args.device, dtype=dtype, enabled=args.precision == 'bf16'):
                _, gen_losses = model.train_step(b, criteria[1], optimizer_idx=1)
            if not torch.isfinite(gen_losses['loss']): raise FloatingPointError('nonfinite generator')
            gen_losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(gen_parameters, 1000., error_if_nonfinite=True)
            optimizers[1].step()
            for p in disc_parameters: p.requires_grad_(True)
            if args.device == 'cuda': torch.cuda.synchronize()
            changed = not torch.equal(before, probe.detach())
            if not changed: raise RuntimeError('sampled generator weights did not update')
            step += 1; cursor += 1
            event(args.run, 'update', step=step, epoch=epoch, cursor=cursor, item_id=row['id'],
                  seconds=time.perf_counter() - started, weight_update=changed,
                  generator_loss=float(gen_losses['loss'].detach()), discriminator_loss=float(disc_losses['loss'].detach()),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None)
            # Drop cache to release the finished autograd graph between steps.
            model.model_outputs_cache = None
            del b, gen_losses, disc_losses
            if args.until_step and step >= args.until_step:
                save('last.pt'); text_probes('last'); event(args.run, 'pilot_boundary', step=step, epoch=epoch); return
            if time.monotonic() - last_save >= args.checkpoint_seconds: save('last.pt')
        if stop: break
        score = validation()
        for scheduler in schedulers: scheduler.step()
        if score < best * .995:
            best, stale = score, 0; save('best.pt'); text_probes('best')
        else:
            stale += 1
        if stale >= args.patience:
            if not reduced:
                for optimizer in optimizers:
                    for group in optimizer.param_groups: group['lr'] *= .3
                reduced, stale = True, 0
                event(args.run, 'lr_reduced', step=step, epoch=epoch)
            else:
                save('last.pt'); text_probes('last'); event(args.run, 'converged', step=step, epoch=epoch, best_validation=best); return
        save('last.pt')
        if args.stage == 'language':
            text_probes(f'epoch-{epoch:04d}',all_cases=True)
            # Every epoch is synthesized, with bounded disk retention. Numeric
            # history remains in events; keep latest six audio snapshots for ASR.
            import shutil
            older=sorted((args.run/'probes').glob('epoch-*'))[:-6]
            for directory in older:
                if directory.is_dir() and directory.parent == args.run/'probes':shutil.rmtree(directory)
        elif epoch % 3 == 0: text_probes('last')
    save('last.pt'); event(args.run, 'interrupted_safely', step=step, epoch=epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['infer','reconstruct','train','check'], required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepared', type=Path, default=DATA)
    parser.add_argument('--stage',choices=['target','language'],default='target')
    parser.add_argument('--init-checkpoint',type=Path,help='New stage: strict generator-only load, fresh critic and training state')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--gpu-authorized', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--text', default='Olá, a pressão está normal.')
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--until-step', type=int, default=0)
    parser.add_argument('--checkpoint-seconds', type=int, default=1800)
    parser.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    parser.add_argument('--lr-gen', type=float, default=2e-5)
    parser.add_argument('--lr-disc', type=float, default=2e-5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--allow-cpu-train', action='store_true')
    args = parser.parse_args(); os.umask(0o077)
    args.run.mkdir(parents=True, mode=0o700, exist_ok=True)
    started = time.perf_counter()
    torch, np, model, config, normalizer = bootstrap(args)
    atomic_json(args.run/'load.json', dict(strict_generator_load=True, base_sha256=BASE_SHA,
                                          parameters=sum(p.numel() for p in model.parameters()),
                                          load_seconds=time.perf_counter()-started, device=args.device,
                                          runtime='coqui-tts0.28.0', torch=torch.__version__))
    if args.action == 'infer': infer(args, torch, np, model, normalizer)
    elif args.action == 'reconstruct': reconstruct(args, torch, np, model, normalizer)
    elif args.action == 'train': train(args, torch, np, model, config, normalizer)
    else:
        rows, fingerprint = rows_and_fingerprint(args.prepared, model, normalizer)
        atomic_json(args.run/'check.json', dict(dataset_fingerprint=fingerprint, counts={k:len(v) for k,v in rows.items()},
                                               tokenizer_train_validation_oov=0, final_test_audio_untouched=True))
        event(args.run, 'check_complete', dataset=fingerprint)


if __name__ == '__main__':
    main()
