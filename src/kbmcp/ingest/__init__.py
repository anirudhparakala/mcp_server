"""kbmcp corpus ingest (dev-only, [corpus] extra)."""

# Docling's PDF pipeline (RT-DETR layout + TableFormer) triggers torch.compile /
# TorchInductor, which JIT-compiles CPU kernels and needs a C++ compiler (MSVC
# cl.exe) that is frequently absent — notably on Windows. Force eager mode so
# parsing works everywhere. Safe here: the parsed DoclingDocument is committed to
# corpus/parsed/ (compile speedup is a one-time-build concern, irrelevant), and
# eager execution is in fact more deterministic. Must be set before torch is
# imported by the parse/chunk submodules; this package __init__ always runs first.
import os as _os

_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
