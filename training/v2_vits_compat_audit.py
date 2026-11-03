"""Compare actual old/new Coqui grapheme classes and tokenizers without Torch.

Execute source AST class/function definitions only; skip package imports and
network/GPU/model loading. Receipts include exact tokens for public phrases.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module = ast.Module(body=[future,*nodes],type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module,str(path),'exec'),namespace)


def frontend(base, config):
    ns = dict(re=re,normalize=unicodedata.normalize,_whitespace_re=re.compile(r'\s+'),
              _pad='<PAD>',_eos='<EOS>',_bos='<BOS>',_blank='<BLNK>',
              _characters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',_punctuations="!'(),-.:;? ",
              logger=logging.getLogger('compat'),normalize_language=lambda language:language)
    definitions(base/'characters.py',{'BaseVocabulary','BaseCharacters','Graphemes'},ns)
    characters,_ = ns['Graphemes'].init_from_config(SimpleNamespace(characters=config['characters']))
    definitions(base/'cleaners.py',{'lowercase','normalize_unicode','replace_symbols','remove_aux_symbols',
                                   'collapse_whitespace','multilingual_cleaners'},ns)
    definitions(base/'tokenizer.py',{'TTSTokenizer'},ns)
    tokenizer = ns['TTSTokenizer'](use_phonemes=config['use_phonemes'],text_cleaner=ns['multilingual_cleaners'],
                                   characters=characters,add_blank=config['add_blank'],
                                   use_eos_bos=config['enable_eos_bos_chars'])
    return tokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/var/tmp/aegis-v2-work/vits'))
    a=p.parse_args();root=a.root
    config=json.loads((root/'model/tts_models--pt--cv--vits/config.json').read_text())
    paths=[root/'source/TTS/tts/utils/text',root/'deps/TTS/tts/utils/text']
    old,new=[frontend(path,config) for path in paths]
    from training.f5_refine_text import normalize_text
    corpus=Path(__file__).resolve().parents[1]/'benchmarks/v2/validation.pt-BR.json'
    cases=json.loads(corpus.read_text())
    results=[]
    for case in cases:
        effective=normalize_text(case['text'])['tts_text']
        x=old.text_to_ids(effective,language='pt');y=new.text_to_ids(effective,language='pt')
        results.append(dict(id=case['id'],source_text=case['text'],effective_text=effective,
                            old_ids=x,new_ids=y,equal=x==y,
                            old_cleaned=old.text_cleaner(effective),new_cleaned=new.text_cleaner(effective,lang='pt')))
    selected={k:config['model_args'].get(k) for k in ('inference_noise_scale','inference_noise_scale_dp','length_scale',
                                                     'noise_scale','noise_scale_dp','use_sdp','condition_dp_on_speaker',
                                                     'use_speaker_embedding','use_language_embedding','max_inference_len')}
    report=dict(old_source='coqui-ai/TTS v0.8.0 @946afa8197f2624fdeca148f6272560d72216488',
                new_source='coqui-tts wheel0.28.0',execution='Actual class/function AST definitions, package imports excluded; no Torch/GPU/model loads',
                source_hashes={str(path/name):digest(path/name) for path in paths for name in ('characters.py','tokenizer.py','cleaners.py')},
                config_sha256=digest(root/'model/tts_models--pt--cv--vits/config.json'),
                vocab_equal=old.characters.vocab==new.characters.vocab,old_vocab=old.characters.vocab,new_vocab=new.characters.vocab,
                specials=dict(pad=old.pad_id,blank=old.blank_id,bos=old.characters.char_to_id(old.characters.bos),eos=old.characters.char_to_id(old.characters.eos)),
                flags={k:config[k] for k in ('add_blank','enable_eos_bos_chars','use_phonemes','text_cleaner')},
                character_config={k:config['characters'][k] for k in ('characters_class','is_unique','is_sorted')},
                inference_config=selected,seed_policy='manual_seed40733133+rep in both baseline matrix and probes; training_seed54321 is not an inference override',
                cleaner_difference='0.28 adds NFC and passes lang=pt; 0.8 multilingual cleaner uses lang=None. No difference for the seven normalized public phrases; & would differ.',
                cases=results,all_case_ids_equal=all(r['equal'] for r in results))
    output=root/'compatibility-audit.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');output.chmod(0o600)
    print(json.dumps(dict(vocab_equal=report['vocab_equal'],vocab_size=len(old.characters.vocab),
                         all_case_ids_equal=report['all_case_ids_equal'],specials=report['specials'],flags=report['flags'],
                         inference_config=selected,receipt=str(output))))
    if not(report['vocab_equal'] and report['all_case_ids_equal']):raise SystemExit(1)


if __name__=='__main__':main()
