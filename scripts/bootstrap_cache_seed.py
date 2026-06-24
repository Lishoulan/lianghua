from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "results" / "stock_cache.duckdb"
DEFAULT_URL = os.getenv(
    "CACHE_SEED_URL",
    "https://github.com/Lishoulan/lianghua/releases/download/cache-seed-v1/stock_cache.duckdb",
)


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb", dir=str(output.parent)) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(url, timeout=180) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_cache_seed(output: Path, url: str, min_bytes: int) -> bool:
    if output.exists() and output.stat().st_size >= min_bytes:
        print(f"[cache-seed] existing cache is ready: {output} ({output.stat().st_size} bytes)")
        return False

    print(f"[cache-seed] downloading seed from {url}")
    _download(url, output)
    size = output.stat().st_size if output.exists() else 0
    if size < min_bytes:
        raise RuntimeError(f"downloaded cache seed is too small: {size} bytes")

    print(f"[cache-seed] downloaded seed: {output} ({size} bytes)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the DuckDB cache from a GitHub release asset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--min-bytes", type=int, default=50 * 1024 * 1024)
    args = parser.parse_args()
    ensure_cache_seed(args.output, args.url, args.min_bytes)


if __name__ == "__main__":
    main()
