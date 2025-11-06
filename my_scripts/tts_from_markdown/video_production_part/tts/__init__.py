from __future__ import annotations

from .ms_tts import (
    AzureSynthOptions,
    AzureTTSResult,
    synthesize_with_azure,
    synthesize_with_azure_ssml,
    split_text,
)
from .validation import normalize_text, validate_segments, correct_segments_text

__all__ = [
    "AzureSynthOptions",
    "AzureTTSResult",
    "synthesize_with_azure",
    "synthesize_with_azure_ssml",
    "split_text",
    "normalize_text",
    "validate_segments",
    "correct_segments_text",
]

