#!/usr/bin/env python3
"""
Fetch Pinned Hugging Face Hub Artifacts.

Downloads the dataset (data/) and training outputs (outputs/) from the
Hugging Face Hub revisions pinned in scripts/hf_artifacts.yaml and verifies
them. An artifact is fetched in one of two ways, chosen by its manifest entry:

- archives: per-sequence tar files. Each tar is checked against the sha256 in
  the manifest, extracted into local_dir, and deleted. A per-file sha256
  record of what was extracted is kept under <local_dir>/.cache/, so later
  runs skip intact sequences and --verify_only works without the tars.
- files: the repo is mirrored into local_dir, and every file is checked
  against the hashes the Hub records for the pinned revision (sha256 for LFS
  files, git blob sha1 for regular files) plus any pinned_sha256 entries.

Both repositories are private: log in once with `hf auth login`, or export
HF_TOKEN (a read-only token is enough). If a transfer stalls without
progress, rerun with HF_HUB_DISABLE_XET=1 to use plain HTTP downloads.

Usage:
    # Fetch and verify everything listed in the manifest
    python scripts/fetch_artifacts.py

    # Fetch only the test sequence
    python scripts/fetch_artifacts.py data --include uuv3.tar

    # Fetch only the training run (checkpoint and logs)
    python scripts/fetch_artifacts.py outputs --include "train/*"

    # Verify local files without downloading
    python scripts/fetch_artifacts.py --verify_only

Author: Luhang Hong
"""

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(os.environ.get('FETCH_ARTIFACTS_CONFIG',
                                     REPO_ROOT / 'scripts' / 'hf_artifacts.yaml'))
STATE_SUBDIR = Path('.cache') / 'fetch_artifacts'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch and verify pinned Hugging Face Hub artifacts')
    parser.add_argument('names', nargs='*',
                        help='Artifact names from the manifest (default: all)')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG),
                        help='Manifest YAML (default: %(default)s)')
    parser.add_argument('--root', type=str, default=str(REPO_ROOT),
                        help='Directory that local_dir entries are relative to '
                             '(default: %(default)s)')
    parser.add_argument('--include', type=str, nargs='+', default=None,
                        help='Only fetch and verify repo paths matching these glob patterns')
    parser.add_argument('--verify_only', action='store_true',
                        help='Verify local files without downloading')
    return parser.parse_args()


def is_sha(value, length):
    return len(value) == length and all(c in '0123456789abcdef' for c in value)


def load_manifest(path):
    """Load the manifest and reject revisions or archive hashes that are not full digests."""
    with open(path, 'r', encoding='utf-8') as f:
        artifacts = (yaml.safe_load(f) or {}).get('artifacts') or {}
    for name, spec in artifacts.items():
        revision = str(spec.get('hf_revision', ''))
        if not is_sha(revision, 40):
            raise ValueError(f'{name}: hf_revision must be a full 40-character '
                             f'commit SHA, got {revision!r}')
        for archive, digest in (spec.get('archives') or {}).items():
            if not is_sha(str(digest).lower(), 64):
                raise ValueError(f'{name}: archive {archive} needs a full sha256, got {digest!r}')
    return artifacts


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def lfs_sha256(sibling):
    lfs = sibling.lfs
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get('sha256')
    return getattr(lfs, 'sha256', None)


def is_selected(path, ignore_patterns, include_patterns):
    if any(fnmatch.fnmatch(path, p) for p in ignore_patterns):
        return False
    return include_patterns is None or any(fnmatch.fnmatch(path, p) for p in include_patterns)


# ----------------------------------------------------------------------------
# archives: per-sequence tar files
# ----------------------------------------------------------------------------

def safe_member_path(member, prefix):
    """Return the member's relative path, or raise if it could escape local_dir."""
    name = PurePosixPath(member.name)
    if name.is_absolute() or '..' in name.parts or name.parts[:1] != (prefix,):
        raise ValueError(f'unsafe member path in archive: {member.name!r}')
    if not (member.isfile() or member.isdir()):
        raise ValueError(f'unsupported member type in archive: {member.name!r}')
    return Path(*name.parts)


def extract_archive(archive_path, local_dir, prefix):
    """Extract a tar into local_dir and return {relative path: [size, sha256]}."""
    record = {}
    with tarfile.open(archive_path, 'r') as tar:
        for member in tar:
            rel = safe_member_path(member, prefix)
            target = local_dir / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tar.extractfile(member) as src, open(target, 'wb') as dst:
                for chunk in iter(lambda: src.read(1 << 20), b''):
                    digest.update(chunk)
                    dst.write(chunk)
            record[rel.as_posix()] = [member.size, digest.hexdigest()]
    return record


def check_record(local_dir, record):
    """Compare extracted files with a stored record; return failure messages."""
    failures = []
    for rel, (size, digest) in sorted(record['files'].items()):
        path = local_dir / rel
        if not path.is_file():
            failures.append(f'missing: {rel}')
        elif path.stat().st_size != size or file_sha256(path) != digest:
            failures.append(f'content mismatch: {rel}')
    return failures


def process_archives(spec, local_dir, include_patterns, verify_only):
    """Fetch (unless verify_only) and verify the archives of one artifact.

    Returns:
        (number of archives checked, list of failure messages)
    """
    state_dir = local_dir / STATE_SUBDIR
    download_dir = state_dir / 'downloads'
    checked = 0
    failures = []

    for archive, expected in spec['archives'].items():
        if not is_selected(archive, [], include_patterns):
            continue
        checked += 1
        expected = str(expected).lower()
        prefix = archive[:-len('.tar')] if archive.endswith('.tar') else archive
        record_path = state_dir / f'{archive}.json'

        problems = ['no fetch record']
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding='utf-8'))
            if record.get('sha256') != expected:
                problems = ['fetch record is for a different archive version']
            else:
                problems = check_record(local_dir, record)
        if not problems:
            print(f'  {archive}: OK ({len(record["files"])} files)')
            continue
        if verify_only:
            failures.extend(f'{archive}: {p}' for p in problems[:20])
            continue

        print(f'  {archive}: downloading ({problems[0]})')
        path = Path(hf_hub_download(
            repo_id=spec['hf_repo'],
            repo_type=spec['hf_repo_type'],
            revision=spec['hf_revision'],
            filename=archive,
            local_dir=str(download_dir),
            token=os.environ.get('HF_TOKEN'),
        ))
        actual = file_sha256(path)
        if actual != expected:
            path.unlink()
            failures.append(f'{archive}: sha256 mismatch after download')
            continue
        files = extract_archive(path, local_dir, prefix)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps({'sha256': expected, 'files': files}),
                               encoding='utf-8')
        path.unlink()
        print(f'  {archive}: extracted {len(files)} files')

    shutil.rmtree(download_dir, ignore_errors=True)
    return checked, failures


# ----------------------------------------------------------------------------
# files: mirror the repo and verify against Hub hashes
# ----------------------------------------------------------------------------

def process_files(spec, local_dir, include_patterns, verify_only):
    """Fetch (unless verify_only) and verify a mirrored repo.

    Returns:
        (number of files checked, list of failure messages)
    """
    ignore_patterns = spec.get('ignore_patterns') or []
    if not verify_only:
        snapshot_download(
            repo_id=spec['hf_repo'],
            repo_type=spec['hf_repo_type'],
            revision=spec['hf_revision'],
            local_dir=str(local_dir),
            allow_patterns=include_patterns,
            ignore_patterns=ignore_patterns,
            token=os.environ.get('HF_TOKEN'),
        )

    info = HfApi().repo_info(spec['hf_repo'], repo_type=spec['hf_repo_type'],
                             revision=spec['hf_revision'], files_metadata=True)
    sha256_cache = {}
    failures = []
    checked = 0

    for sibling in info.siblings:
        rel = sibling.rfilename
        if not is_selected(rel, ignore_patterns, include_patterns):
            continue
        checked += 1
        path = local_dir / rel
        if not path.is_file():
            failures.append(f'missing: {rel}')
            continue
        if sibling.size is not None and path.stat().st_size != sibling.size:
            failures.append(f'size mismatch: {rel}')
            continue
        expected = lfs_sha256(sibling)
        if expected:
            sha256_cache[rel] = file_sha256(path)
            if sha256_cache[rel] != expected:
                failures.append(f'sha256 mismatch: {rel}')
        elif git_blob_sha1(path) != sibling.blob_id:
            failures.append(f'git blob mismatch: {rel}')

    for rel, expected in (spec.get('pinned_sha256') or {}).items():
        if not is_selected(rel, [], include_patterns):
            continue
        path = local_dir / rel
        if not path.is_file():
            failures.append(f'pinned file missing: {rel}')
            continue
        actual = sha256_cache.get(rel) or file_sha256(path)
        if actual != str(expected).lower():
            failures.append(f'pinned sha256 mismatch: {rel}')

    return checked, failures


def main():
    args = parse_args()
    artifacts = load_manifest(args.config)
    names = args.names or list(artifacts)
    unknown = [n for n in names if n not in artifacts]
    if unknown:
        sys.exit(f'Unknown artifact(s): {unknown}. Manifest lists: {list(artifacts)}')

    root = Path(args.root)
    all_ok = True
    for name in names:
        spec = artifacts[name]
        local_dir = root / spec['local_dir']
        action = 'Verifying' if args.verify_only else 'Fetching'
        print(f'[{name}] {action} {spec["hf_repo"]} @ {spec["hf_revision"]} -> {local_dir}')
        process = process_archives if spec.get('archives') else process_files
        checked, failures = process(spec, local_dir, args.include, args.verify_only)
        if checked == 0:
            failures.append('no files matched')
        if failures:
            all_ok = False
            print(f'[{name}] FAILED: {len(failures)} problem(s) in {checked} checked')
            for message in failures[:20]:
                print(f'  {message}')
            if len(failures) > 20:
                print(f'  ... and {len(failures) - 20} more')
        else:
            print(f'[{name}] OK: {checked} checked')

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
