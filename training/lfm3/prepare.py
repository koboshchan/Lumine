"""Download the hash-bound causal annotation index from the Lumine-only bucket."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_registry(root, config):
    from huggingface_hub import HfApi
    api = HfApi()
    bucket = 'koboshchan/checkpoints'
    if not api.bucket_info(bucket).private:
        raise ValueError('Expected private Lumine bucket')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = 'lumine/immutable-payload-v1'
    def fetch(key, dest, expected):
        if dest.exists() and sha(dest) == expected:
            return dest
        tmp = dest.with_suffix('.partial')
        api.download_bucket_files(bucket, [(key, tmp)], raise_on_missing_files=True)
        if sha(tmp) != expected:
            raise ValueError('Pinned annotation payload hash mismatch')
        tmp.replace(dest)
        return dest
    manifest_sha = config['payload_manifest_sha256']
    manifest = json.loads(fetch(f'{prefix}/manifests/{manifest_sha}.json',
                                root / 'annotation-manifest.json', manifest_sha).read_text())
    matches = [r for r in manifest['files'] if r['sha256'] == config['index_sha256']]
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous canonical causal registry')
    index = fetch(matches[0]['key'], root / 'expanded.sqlite', config['index_sha256'])
    remote_sha = '520b1c01c7123a64a6f5f15a1e4e93fafdd7b0a1b8e085e6eb056f68cbd5073b'
    remote = json.loads(fetch(f'{prefix}/objects/{remote_sha}', root / 'source-config.json', remote_sha).read_text())
    if len(remote['shards']) != 94:
        raise ValueError('Unexpected source inventory')
    for row in remote['shards']:
        path = root / row['file']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Dataset shard hash mismatch')
    return index
