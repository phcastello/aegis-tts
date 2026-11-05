"""Portable contracts for the real optional-reference delivery CLI.

Only synthetic temporary files are used. Runtime config construction stops
before artifact loading, service startup, or any neural operation.
"""
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from training import v2_voxcpm_delivery_launcher as launcher


FRONTENDS = {
    "original-nfc": "OriginalNFC",
    "f5-refinement-1": "F5Refinement1",
    "ptbr-numbers-v1": "PtbrNumbers1",
}


def digest(values):
    return hashlib.sha256(values).hexdigest()


def serialized(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


class ReferenceCLIContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.reference = self.root / "reference.wav"
        with wave.open(str(self.reference), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(48000)
            stream.writeframes(b"\x00\x00" * 8)
        self.reference_sha = digest(self.reference.read_bytes())

    def args(self, frontend="original-nfc", **updates):
        args = SimpleNamespace(
            python=self.root / "env/bin/python", run=self.root / "run",
            checkpoint=self.root / "selected.pt", checkpoint_sha256=digest(b"checkpoint"),
            source=self.root / "source", model=self.root / "model",
            cache=self.root / "cache", overlay=self.root / "overlay",
            compiler_prerequisites=self.root / "compiler.json", port=8002,
            ready_timeout=600., frontend=frontend, startup_ms=320., timeout=180.,
            reference_wav=None, reference_sha256=None, action="plan",
            candidate="synthetic", repetitions=3, gpu_authorized=False,
            final_test=False, decision=None, corpus=None, spoken=None,
        )
        vars(args).update(updates)
        return args

    def legacy_profile(self, frontend):
        return f"AegisVoicev2.0-VoxCPM2-LoRA-r8-N10-Passthrough-{FRONTENDS[frontend]}-Experimental"

    def legacy_config(self, args):
        # Frozen pre-reference configuration shape, including insertion order.
        return dict(source=str(args.source), model=str(args.model),
                    adapter=str(args.checkpoint), adapter_sha256=args.checkpoint_sha256,
                    cache=str(args.cache), gpu_lock=str(args.run.parent / "gateway-gpu.lock"),
                    compiler_prerequisites=str(args.compiler_prerequisites),
                    dsp="passthrough", frontend=args.frontend,
                    instrument_replays=True, seed=40733133)

    def legacy_settings(self, args):
        # The legacy plan payload has no reference keys, even with null values.
        return dict(
            endpoint="http://127.0.0.1:8002/v1/audio/speech",
            voice_profile=self.legacy_profile(args.frontend), frontend=args.frontend,
            model_sha256=args.checkpoint_sha256, startup_ms=320.,
            voice_profile_checked_in="Every speech response/completion, not readiness",
            startup_evidence=str(args.run / "startup-real.json"), timeout=180.,
            backend="voxcpm2",
            internal_warmups="Startup receipt must record actual backend boot warmups separately",
            seed=40733133, payload=dict(stream=True, response_format="pcm"),
            scope="Client HTTP PCM; common receive/onset/FIFO rules; no hardware playback delay; no server lifecycle",
            launcher=dict(
                python=str(args.python), source=str(args.source), model=str(args.model),
                checkpoint=str(args.checkpoint), overlay=str(args.overlay), cache=str(args.cache),
                compiler_prerequisites=str(args.compiler_prerequisites), startup_ms=320.,
                ready_timeout=600., lifecycle="Worker→launcher; launcher owns uvicorn child and directly calls client.execute in same PID; always stops owned server",
            ),
        )

    def settings(self, args):
        with patch.object(launcher.protocol, "make_plan",
                          side_effect=lambda _args, _backend, settings, _sources: copy.deepcopy(settings)) as make:
            result = launcher.build_plan(args)
        self.assertEqual(make.call_args.args[1], "voxcpm2-http")
        return result

    def capture_config(self, args):
        class ConfigCaptured(Exception):
            pass

        saved = []

        def save(path, value):
            self.assertEqual(path, args.run / "runtime.private.json")
            saved.append(copy.deepcopy(value))

        def bindings(seen, path):
            self.assertIs(seen, args)
            self.assertEqual(path, args.run / "runtime.private.json")
            self.assertEqual(len(saved), 1)
            raise ConfigCaptured()

        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher.protocol, "owner_guard", return_value={}))
            stack.enter_context(patch.object(launcher.protocol, "gpu_set"))
            stack.enter_context(patch.object(launcher.protocol, "budget", return_value={}))
            stack.enter_context(patch.object(launcher.gateway, "verify_overlay"))
            stack.enter_context(patch.object(launcher.gateway, "cpu_check"))
            stack.enter_context(patch.object(launcher.protocol, "read", return_value={"sha256": args.checkpoint_sha256}))
            stack.enter_context(patch.object(launcher.protocol, "save", side_effect=save))
            stack.enter_context(patch.object(launcher, "artifact_bindings", side_effect=bindings))
            stack.enter_context(patch.object(launcher.socket, "socket"))
            spawn = stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("spawn forbidden")))
            with self.assertRaises(ConfigCaptured):
                launcher.execute(args, {})
            spawn.assert_not_called()
        return saved[0]

    def test_none_matches_legacy_profile_config_bytes_and_plan_all_frontends(self):
        for frontend in FRONTENDS:
            with self.subTest(frontend=frontend):
                args = launcher.decorate(self.args(frontend))
                self.assertEqual(args.profile, self.legacy_profile(frontend))
                self.assertIsNone(args.reference_wav)
                self.assertIsNone(args.reference_sha256)
                config = self.capture_config(args)
                path = self.root / "captured-config.json"
                launcher.protocol.save(path, config)
                self.assertEqual(path.read_bytes(), serialized(self.legacy_config(args)))
                self.assertEqual(serialized(self.settings(args)), serialized(self.legacy_settings(args)))

    def test_legacy_callers_without_reference_attributes_keep_none_contract(self):
        args = self.args()
        del args.reference_wav
        del args.reference_sha256
        decorated = launcher.decorate(args)
        self.assertIsNone(decorated.reference_wav)
        self.assertIsNone(decorated.reference_sha256)
        self.assertEqual(self.capture_config(decorated), self.legacy_config(decorated))
        self.assertEqual(self.settings(decorated), self.legacy_settings(decorated))

    def test_reference_pair_absolute_path_and_lowercase_sha_are_required(self):
        invalid = [
            dict(reference_wav=self.reference), dict(reference_sha256=self.reference_sha),
            dict(reference_wav=Path("relative.wav"), reference_sha256=self.reference_sha),
            dict(reference_wav=str(self.reference), reference_sha256=self.reference_sha),
        ]
        invalid.extend(dict(reference_wav=self.reference, reference_sha256=value)
                       for value in ("A" * 64, "0" * 63, "0" * 65, "g" * 64, 4))
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                launcher.decorate(self.args(**values))

    def test_reference_changes_profile_and_reaches_plan_and_runtime_all_frontends(self):
        for frontend in FRONTENDS:
            with self.subTest(frontend=frontend):
                args = launcher.decorate(self.args(frontend, reference_wav=self.reference,
                                                   reference_sha256=self.reference_sha))
                self.assertEqual(args.profile, self.legacy_profile(frontend) + "-AcousticReference")
                self.assertNotIn(str(self.reference), args.profile)
                self.assertNotIn(self.reference_sha, args.profile)
                config = self.capture_config(args)
                expected = self.legacy_config(args)
                expected.update(reference_wav=str(self.reference), reference_sha256=self.reference_sha)
                self.assertEqual(config, expected)
                settings = self.settings(args)
                self.assertEqual(settings["voice_profile"], args.profile)
                self.assertEqual(settings["launcher"]["reference_wav"], str(self.reference))
                self.assertEqual(settings["launcher"]["reference_sha256"], self.reference_sha)

    def test_real_artifact_bindings_pin_reference_and_reject_changed_bytes(self):
        args = launcher.decorate(self.args(reference_wav=self.reference, reference_sha256=self.reference_sha))
        args.model.mkdir()
        args.checkpoint.write_bytes(b"checkpoint")
        base = b"synthetic base"
        vae = b"synthetic vae"
        (args.model / "model.safetensors").write_bytes(base)
        (args.model / "audiovae.pth").write_bytes(vae)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "tokenization_voxcpm2.py"):
            (args.model / name).write_bytes(b"synthetic metadata")
        config = self.root / "runtime.json"
        config.write_text("{}")
        with patch.object(launcher, "BASE_SHA", digest(base)), patch.object(launcher, "VAE_SHA", digest(vae)):
            bound = launcher.artifact_bindings(args, config)
            self.assertEqual([x for x in bound if x["role"] == "acoustic_reference"],
                             [dict(role="acoustic_reference", path=str(self.reference), sha256=self.reference_sha)])
            without = launcher.decorate(self.args())
            self.assertNotIn("acoustic_reference", [x["role"] for x in launcher.artifact_bindings(without, config)])
            self.reference.write_bytes(b"changed after pinning")
            with self.assertRaises(ValueError):
                launcher.artifact_bindings(args, config)

    def test_real_cli_defaults_frontends_and_reference_pair_forwarding(self):
        prefix = ["launcher", "--run", str(self.root / "cli-run"), "--candidate", "synthetic",
                  "--checkpoint", str(self.root / "selected.pt"), "--checkpoint-sha256", digest(b"checkpoint")]
        cases = [([], "ptbr-numbers-v1", False)]
        for frontend in FRONTENDS:
            for reference in (False, True):
                flags = ["--frontend", frontend]
                if reference:
                    flags += ["--reference-wav", str(self.reference), "--reference-sha256", self.reference_sha]
                cases.append((flags, frontend, reference))
        for flags, frontend, reference in cases:
            with self.subTest(flags=flags), patch.object(sys, "argv", prefix + flags), \
                    patch.object(launcher, "build_plan", return_value={}) as plan, \
                    patch.object(launcher.protocol, "plan_or_verify", return_value=False), \
                    patch.object(launcher, "execute", side_effect=AssertionError("execute forbidden")) as execute, \
                    patch.object(launcher.os, "umask"), redirect_stdout(io.StringIO()):
                launcher.main()
                args = plan.call_args.args[0]
                self.assertEqual(args.action, "plan")
                self.assertEqual(args.frontend, frontend)
                self.assertEqual(args.reference_wav, self.reference if reference else None)
                self.assertEqual(args.reference_sha256, self.reference_sha if reference else None)
                execute.assert_not_called()
        for flags in (["--reference-wav", str(self.reference)], ["--reference-sha256", self.reference_sha]):
            with self.subTest(flags=flags), patch.object(sys, "argv", prefix + flags), \
                    patch.object(launcher, "build_plan", side_effect=AssertionError("planning forbidden")), \
                    self.assertRaises(ValueError):
                launcher.main()

    def test_frozen_plan_rejects_reference_toggle_pin_or_path_changes(self):
        args = launcher.decorate(self.args(reference_wav=self.reference, reference_sha256=self.reference_sha))
        plan = dict(created_utc="fixture", candidate=args.candidate, settings=self.settings(args))
        self.assertFalse(launcher.protocol.plan_or_verify(args, plan))
        frozen = (args.run / "plan.json").read_bytes()
        args.action = "execute"
        args.gpu_authorized = True
        self.assertTrue(launcher.protocol.plan_or_verify(args, copy.deepcopy(plan)))
        for update in (dict(reference_wav=None, reference_sha256=None),
                       dict(reference_sha256=digest(b"another reference")),
                       dict(reference_wav=self.root / "another.wav")):
            with self.subTest(update=update):
                changed = copy.deepcopy(args)
                vars(changed).update(update)
                launcher.decorate(changed)
                altered = dict(plan, settings=self.settings(changed))
                with self.assertRaisesRegex(ValueError, "Frozen plan differs"):
                    launcher.protocol.plan_or_verify(changed, altered)
        self.assertEqual((args.run / "plan.json").read_bytes(), frozen)


if __name__ == "__main__":
    unittest.main()
