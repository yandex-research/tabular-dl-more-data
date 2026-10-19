"""Package and verify the pinned Kumo Hugging Face cache for offline jobs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO_ID = 'nvidia/Kumo-Tabular'
REVISION = 'v1.0.0'
SNAPSHOT = 'bd7fa122b516c7355583873ffcf38f5e79403ebd'
SDM_COMMIT = '5d663369ff8ead4a78c910dbb8483d558e46627b'
CACHE_NAME = 'models--nvidia--Kumo-Tabular'
FILENAMES = tuple(
    f'{size}/{task}.pt'
    for size in ('small', 'medium', 'large')
    for task in ('classifier', 'regressor')
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(directory: Path, *, check_hashes: bool = True) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = {
        'repo_id': REPO_ID,
        'revision': REVISION,
        'snapshot': SNAPSHOT,
        'sdm_commit': SDM_COMMIT,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Kumo artifact does not match the pinned model/source')
    if set(manifest['files']) != set(FILENAMES):
        raise ValueError('Kumo artifact must contain all six checkpoints')
    cache = directory / 'hub' / CACHE_NAME
    if (cache / 'refs' / REVISION).read_text().strip() != SNAPSHOT:
        raise ValueError('Kumo cache revision differs from the pinned snapshot')
    for name, digest in manifest['files'].items():
        path = cache / 'snapshots' / SNAPSHOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if check_hashes and sha256(path) != digest:
            raise ValueError(f'Kumo checkpoint checksum differs: {name}')
    return manifest


def prepare(cache_root: Path, output: Path) -> dict:
    """Copy cached weights without network access or host-cache symlinks."""
    cache = cache_root / CACHE_NAME
    if (cache / 'refs' / REVISION).read_text().strip() != SNAPSHOT:
        raise ValueError('Source cache revision differs from the pinned snapshot')
    source = cache / 'snapshots' / SNAPSHOT
    for name in FILENAMES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'hub' / CACHE_NAME
    hashes = {}
    for name in FILENAMES:
        original = source / name
        # HF blob names for these LFS checkpoints are their SHA-256 hashes.
        digest = sha256(original)
        if original.is_symlink() and original.resolve().name != digest:
            raise ValueError(f'Corrupt source checkpoint: {name}')
        copied = destination / 'snapshots' / SNAPSHOT / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
        hashes[name] = digest
    refs = destination / 'refs'
    refs.mkdir(parents=True)
    (refs / REVISION).write_text(SNAPSHOT)
    manifest = {
        'repo_id': REPO_ID,
        'revision': REVISION,
        'snapshot': SNAPSHOT,
        'sdm_commit': SDM_COMMIT,
        'files': hashes,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='action', required=True)
    pack = subparsers.add_parser('prepare')
    pack.add_argument('--cache-root', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    check = subparsers.add_parser('verify')
    check.add_argument('directory', type=Path)
    args = parser.parse_args()
    manifest = (
        prepare(args.cache_root, args.output)
        if args.action == 'prepare'
        else verify(args.directory)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
