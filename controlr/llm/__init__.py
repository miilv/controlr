"""LLM layer: streaming client, cache-marker placement, append-only transcript,
and a scripted fake for offline tests. See docs/ARCHITECTURE.md "LLM client",
"Caching", "Transcript"."""

from .caching import apply_cache_markers, cache_style_for
from .client import LLMClient, LLMResult, Timings, Usage, normalize_usage
from .fake import FakeLLM
from .transcript import ImagePart, Transcript, encode_image

__all__ = [
    "LLMClient", "LLMResult", "Timings", "Usage", "normalize_usage",
    "apply_cache_markers", "cache_style_for",
    "ImagePart", "Transcript", "encode_image",
    "FakeLLM",
]
