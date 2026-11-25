from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = REPO_ROOT / "sample" / "sample-data-documents"


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    if not SAMPLE_DOCS.is_dir() or not any(SAMPLE_DOCS.iterdir()):
        pytest.skip("sample documents are not present in this checkout")
    return SAMPLE_DOCS
