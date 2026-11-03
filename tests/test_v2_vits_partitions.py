"""Regression: an appended transcript correction must not duplicate a clip."""
import unittest
from training.v2_vits import validate_partitions


class AppendedTranscriptCorrection(unittest.TestCase):
    def test_same_id_is_rejected_for_conflicting_and_identical_corrections(self):
        # Public-corpus audit found sample-47 appended with unrelated text and
        # sample-2142 appended twice identically. Both would overweight a clip.
        for identifier,texts in [('sample-47',('Olá, Daniela!','Contudo, Jack.')),
                                 ('sample-2142',('Uma frase.','Uma frase.'))]:
            with self.subTest(identifier=identifier):
                rows={s:[dict(id=s,text=s,group_id=s,sha256=s)] for s in ('train','validation','test')}
                rows['train']=[dict(id=identifier,text=text,group_id='source-batch-0',sha256='same-audio') for text in texts]
                with self.assertRaisesRegex(ValueError,'duplicate ID'):
                    validate_partitions(rows)


if __name__=='__main__':unittest.main()
