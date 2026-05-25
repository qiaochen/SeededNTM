"""Local caching and downloading utilities for offline database files."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "marker_agent"
)


def get_cache_dir(cache_dir: Optional[str] = None) -> Path:
    """Return the cache directory, creating it if necessary."""
    path = Path(cache_dir) if cache_dir else Path(DEFAULT_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_download(
    url: str,
    filename: str,
    cache_dir: Optional[str] = None,
    expected_md5: Optional[str] = None,
) -> Path:
    """Download a file to cache if not already present.

    Returns the local path to the cached file.
    """
    dest_dir = get_cache_dir(cache_dir)
    dest_path = dest_dir / filename

    if dest_path.exists():
        if expected_md5 and _md5(dest_path) != expected_md5:
            logger.warning("MD5 mismatch for %s, re-downloading", filename)
        else:
            return dest_path

    logger.info("Downloading %s -> %s", url, dest_path)
    try:
        urlretrieve(url, str(dest_path))
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        raise

    if expected_md5 and _md5(dest_path) != expected_md5:
        logger.warning("MD5 mismatch after download for %s", filename)

    return dest_path


def _md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_local_file(
    filename: str,
    search_dirs: Optional[list] = None,
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """Search for a local database file in standard locations."""
    dirs_to_check = []
    if search_dirs:
        dirs_to_check.extend(search_dirs)
    dirs_to_check.append(str(get_cache_dir(cache_dir)))
    dirs_to_check.append(os.getcwd())
    dirs_to_check.append(os.path.join(os.getcwd(), "data"))

    for d in dirs_to_check:
        p = Path(d) / filename
        if p.exists():
            return p
    return None
