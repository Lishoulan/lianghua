from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

from scripts.free_data_sources import patch_akshare_with_efinance_fallback


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_TAG = sys.implementation.cache_tag
ENV_FILE = Path(os.getenv("DOCKER_ENV_FILE", REPO_ROOT / ".env.docker.local"))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _source_path_for(fullname: str) -> Path:
    return REPO_ROOT.joinpath(*fullname.split(".")).with_suffix(".py")


def _package_path_for(fullname: str) -> Path:
    return REPO_ROOT.joinpath(*fullname.split("."), "__init__.py")


def _pyc_path_for(source_path: Path) -> Path:
    return source_path.parent / "__pycache__" / f"{source_path.stem}.{CACHE_TAG}.pyc"


def _is_broken_source(path: Path) -> bool:
    if not path.exists():
        return False

    data = path.read_bytes()
    if b"\x00" in data:
        return True

    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True

    return False


class PatchedSourcelessFileLoader(importlib.machinery.SourcelessFileLoader):
    def __init__(self, fullname: str, pyc_path: str, source_path: str):
        super().__init__(fullname, pyc_path)
        self.source_path = source_path

    def exec_module(self, module):
        module.__file__ = self.source_path
        module.__cached__ = self.path
        return super().exec_module(module)


class BytecodeFallbackFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        source_path = _source_path_for(fullname)
        package_path = _package_path_for(fullname)

        if package_path.exists():
            pyc_path = _pyc_path_for(package_path)
            if pyc_path.exists() and _is_broken_source(package_path):
                loader = PatchedSourcelessFileLoader(fullname, str(pyc_path), str(package_path))
                spec = importlib.util.spec_from_loader(fullname, loader, origin=str(pyc_path), is_package=True)
                if spec is not None:
                    spec.submodule_search_locations = [str(package_path.parent)]
                return spec

        if source_path.exists():
            pyc_path = _pyc_path_for(source_path)
            if pyc_path.exists() and _is_broken_source(source_path):
                loader = PatchedSourcelessFileLoader(fullname, str(pyc_path), str(source_path))
                return importlib.util.spec_from_loader(fullname, loader, origin=str(pyc_path))

        return None


def _install_import_hook() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.meta_path.insert(0, BytecodeFallbackFinder())


def _disable_broken_dotenv() -> None:
    try:
        import dotenv
    except ImportError:
        return

    def _safe_load_dotenv(*args, **kwargs):  # noqa: ANN002, ANN003
        return False

    dotenv.load_dotenv = _safe_load_dotenv


def _install_source_fallbacks() -> None:
    try:
        patch_akshare_with_efinance_fallback()
    except Exception:
        pass


def run_daily_push() -> None:
    _load_env_file(ENV_FILE)
    _disable_broken_dotenv()
    _install_import_hook()
    _install_source_fallbacks()

    module = importlib.import_module("classic_ta.daily_push")
    entrypoint = getattr(module, "daily_push", None)
    if not callable(entrypoint):
        raise RuntimeError("classic_ta.daily_push does not expose a callable daily_push()")

    entrypoint()


def load_daily_push_module():
    _load_env_file(ENV_FILE)
    _disable_broken_dotenv()
    _install_import_hook()
    _install_source_fallbacks()
    return importlib.import_module("classic_ta.daily_push")


def run_prewarm() -> None:
    module = load_daily_push_module()
    prewarm = getattr(module, "prewarm_data", None)
    if not callable(prewarm):
        raise RuntimeError("classic_ta.daily_push does not expose a callable prewarm_data()")
    prewarm()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bytecode-backed daily_push entrypoint.")
    parser.add_argument("--prewarm-only", action="store_true")
    args = parser.parse_args()
    if args.prewarm_only:
        run_prewarm()
    else:
        run_daily_push()


if __name__ == "__main__":
    main()
