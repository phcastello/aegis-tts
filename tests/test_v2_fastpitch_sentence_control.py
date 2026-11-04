import sys
import unittest
from unittest.mock import patch

from training import v2_fastpitch_sentence_control as control


class Values(list):
    @property
    def size(self):
        return len(self)


class FakeBackend:
    def __init__(self):
        self.calls, self.seeds = [], []
        self.torch = type('Torch', (), {'manual_seed': self.seeds.append})()
        self.np = type('Numpy', (), {'concatenate': staticmethod(lambda rows: Values(v for row in rows for v in row))})()

    def generate(self, text):
        self.calls.append(text)
        return Values([len(self.calls) / 10, -len(self.calls) / 10])


class SentenceControlTests(unittest.TestCase):
    def test_spans_preserve_every_character_and_accents(self):
        text = 'João, confirme a pressão. Não libere a válvula! Qual é a revisão?'
        spans = control.chunks(text)
        self.assertEqual(''.join(spans), text)
        self.assertEqual(len(spans), 3)

    def test_decimal_and_abbreviation_are_preserved(self):
        text = 'O Dr. João mediu 2.7 kPa. Confirme às 14h05.'
        spans = control.chunks(text)
        self.assertEqual(''.join(spans), text)
        self.assertEqual(len(spans), 2)
        self.assertIn('Dr. João', spans[0])
        self.assertIn('2.7', spans[0])

    def test_word_and_character_bounds_refuse_truncation(self):
        text = ' '.join(['pressão'] * 100)
        spans = control.chunks(text)
        self.assertEqual(''.join(spans), text)
        self.assertTrue(all(len(p) <= 180 and len(p.split()) <= 32 for p in spans))
        with self.assertRaises(ValueError):
            control.chunks('a' * 181)

    def test_empty_content_is_not_valid_audio_input(self):
        for text in ('', '   ', '\n'):
            with self.assertRaises(ValueError):
                control.chunks(text)

    def test_adapter_preserves_complete_arrays_without_audio_edits(self):
        backend = control.segmented_backend(FakeBackend)()
        text = 'Primeira frase. Segunda frase.'
        values = backend.generate(text)
        self.assertEqual(values, [.1, -.1, .2, -.2])
        self.assertEqual(''.join(backend.calls), text)
        self.assertEqual(backend.seeds, [control.quality.SEED] * 2)
        self.assertEqual([r['samples'] for r in backend.control_segments], [2, 2])

    def test_native_failure_is_propagated_without_partial_success(self):
        class Failing(FakeBackend):
            def generate(self, text):
                if self.calls:
                    raise ValueError('native failure')
                return super().generate(text)
        with self.assertRaisesRegex(ValueError, 'native failure'):
            control.segmented_backend(Failing)().generate('Primeira. Segunda.')

    def test_aggregate_audio_bound_precedes_concatenation(self):
        class Oversize(FakeBackend):
            def generate(self, text):
                return type('OversizeValues', (), {'size': control.quality.MAX_AUDIO_SECONDS * 24000 + 1})()
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            control.segmented_backend(Oversize)().generate('Frase.')

    def test_missing_gpu_authorization_never_reaches_executor(self):
        with patch.object(sys, 'argv', ['control', '--action', 'execute', '--plan-sha256', 'a' * 64]):
            with patch.object(control, 'execute') as execute:
                with self.assertRaisesRegex(ValueError, 'authorization'):
                    control.main()
                execute.assert_not_called()
        self.assertNotIn('torch', sys.modules)

    def test_complete_span_proof_rejects_missing_words_or_samples(self):
        plan = {'roster': {case: {'spans': ['Frase.']} for case in control.quality.CASE_IDS}}
        rows = [{'case': case, 'candidate': control.CANDIDATE, 'outcome': 'ok', 'samples': 256,
                 'generated_segments': [{'text': 'Frase.', 'samples': 256}]} for case in control.quality.CASE_IDS]
        control.verify_segments(rows, plan)
        rows[-1]['generated_segments'][0]['text'] = 'Outra.'
        with self.assertRaisesRegex(ValueError, 'missing'):
            control.verify_segments(rows, plan)
        rows[-1]['generated_segments'][0]['text'] = 'Frase.'
        rows[-1]['samples'] = 512
        with self.assertRaisesRegex(ValueError, 'missing'):
            control.verify_segments(rows, plan)

    def test_partial_roster_cannot_supply_complete_span_proof(self):
        with self.assertRaisesRegex(ValueError, 'seven-case'):
            control.verify_segments([], {'roster': {}})

    def test_unattempted_slots_keep_control_candidate_for_asr_jobs(self):
        cases = [{'id': case} for case in control.quality.CASE_IDS]
        rows = control.finalize_control_rows([], cases, 'aborted', control.quality.finalize_rows)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(r['candidate'] == control.CANDIDATE and r['outcome'] == 'not_attempted' for r in rows))

    def test_external_budget_counts_whole_cache_and_docker(self):
        cache_bytes, other, docker = 311136256, 3796992, 352256
        threshold = 40_000_000_000 - control.quality.PAIR_EXTRA - cache_bytes - other - docker - control.DOCKER_LOG_RESERVE
        result = control.external_budget({'working_allocated_bytes': threshold}, cache_bytes, other, docker)
        self.assertEqual(result['total_projected_bytes'], 40_000_000_000)
        self.assertEqual(result['cache_exclusion_credit_bytes'], 0)
        with self.assertRaises(RuntimeError):
            control.external_budget({'working_allocated_bytes': threshold + 1}, cache_bytes, other, docker)


if __name__ == '__main__':
    unittest.main()
