"""Original NFC text for the experimental Vox LoRA recipe; no v1 lexicon."""
from __future__ import annotations
import unicodedata
import re
from .normalization import AppliedRule, NormalizationResult
from .segmentation import SpeechSegmenter


class OriginalTextNormalizer:
    def normalize(self,text):
        if not isinstance(text,str) or not text.strip(): raise ValueError('text must not be empty')
        if len(text)>12000: raise ValueError('text exceeds maximum length')
        if any(unicodedata.category(c) in {'Cs','Cc'} and c not in '\n\r\t' for c in text):
            raise ValueError('unsupported text control/surrogate')
        out=unicodedata.normalize('NFC',text)
        rules=[AppliedRule('experimental_original_nfc',1)] if out!=text else []
        return NormalizationResult(text,out,rules,[])


class OriginalTextSegmenter(SpeechSegmenter):
    """Restore opaque technical tokens in every sentence, not only the last.

    The v1 normalizer usually expands/removes these tokens before segmenting;
    original NFC Vox text exposes the parent segmenter's last-fragment lookup
    map assumption. Keep this correction local to the experimental frontend.
    """
    def _boundaries(self,protected):
        maps=''.join(re.findall(r'\x00\d+:.*?\x00',protected))
        return [(re.sub(r'\x00\d+:.*?\x00','',part)+maps,reason)
                for part,reason in super()._boundaries(protected)]


class RefinedTextNormalizer:
    """Explicit F5-refinement-1 control; never the default Vox training frontend."""
    def normalize(self,text):
        from .experimental_refinement import normalize_text
        OriginalTextNormalizer().normalize(text)
        result=normalize_text(text)
        if len(result['tts_text'])>12000:raise ValueError('expanded text exceeds maximum length')
        return NormalizationResult(text,result['tts_text'],
            [AppliedRule(r['id'],r['count']) for r in result['applied_rules']],result['warnings'])
