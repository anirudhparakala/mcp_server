"""Shared pytest fixtures.

`safe_tmp_path` replaces pytest's built-in `tmp_path` fixture project-wide.

Why: on this development machine, `tmp_path` fails with
`PermissionError: [WinError 5] Access is denied:
'C:\\Users\\aniru\\AppData\\Local\\Temp\\pytest-of-aniru'` — a pre-existing
bookkeeping directory under the Windows temp folder is ACL-locked, even for
its owning user, and `tmp_path` must enumerate it for numbered-dir
reuse/cleanup. `tempfile.TemporaryDirectory()` does not touch that
directory and works correctly here, so this fixture wraps it in the same
shape `tmp_path` provides (a `pathlib.Path` to a fresh, empty directory,
cleaned up automatically at the end of the test).
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def safe_tmp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
