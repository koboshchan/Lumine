"""Serialized asynchronous checkpoint archives with verified read-back."""
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bucket_parts(uri):
    prefix = 'hf://buckets/'
    if not uri.startswith(prefix):
        raise ValueError('Expected hf://buckets/namespace/bucket/path.tar')
    pieces = uri[len(prefix):].split('/', 2)
    if len(pieces) != 3:
        raise ValueError('Bucket archive URI needs a file path')
    return '/'.join(pieces[:2]), pieces[2]


def restore(source, output, api):
    output = Path(output)
    if source == 'latest':
        candidates = [p for p in output.glob('checkpoint-*')
                      if p.is_dir() and all((p / n).is_file() for n in
                      ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'training-policy.json'))]
        if not candidates:
            raise ValueError('No complete local checkpoint; provide a bucket .tar URI with --resume')
        return str(max(candidates, key=lambda p: int(p.name.split('-')[-1])))
    if not source.endswith('.tar'):
        return source
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix='.restore-') as tmp:
        archive = Path(source)
        if source.startswith('hf://'):
            bucket, remote = bucket_parts(source)
            archive = Path(tmp) / 'download.tar'
            api.download_bucket_files(bucket, [(remote, str(archive))], raise_on_missing_files=True)
        with tarfile.open(archive, 'r:') as tar:
            members = tar.getmembers()
            roots = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
            if len(roots) != 1:
                raise ValueError('Archive must contain exactly one checkpoint directory')
            name = roots.pop()
            if not name.startswith('checkpoint-') or not name[11:].isdigit():
                raise ValueError('Invalid checkpoint archive root')
            for m in members:
                if Path(m.name).is_absolute() or '..' in Path(m.name).parts or not (m.isfile() or m.isdir()):
                    raise ValueError('Unsafe checkpoint archive entry')
            tar.extractall(tmp, members=members, filter='data')
        checkpoint = Path(tmp) / name
        for required in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'training-policy.json'):
            if not (checkpoint / required).is_file():
                raise ValueError('Incomplete archive: ' + required)
        destination = output / name
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}; use its directory to resume')
        os.replace(checkpoint, destination)
    return str(destination)


class CheckpointUploads:
    def __init__(self, output, bucket, api):
        self.output, self.bucket, self.api = Path(output), bucket.rstrip('/'), api
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-upload')
        self.pending = []
        self.verified = set()

    def submit(self, checkpoint):
        # Called from on_save, after Trainer and processor have closed their files.
        self.pending.append(self.pool.submit(self._upload, Path(checkpoint)))

    def _upload(self, checkpoint):
        try:
            self._pack_upload_verify(checkpoint)
        except Exception as exc:
            print(f'Checkpoint upload FAILED for {checkpoint.name}; local files retained: {exc}', flush=True)
            raise

    def _pack_upload_verify(self, checkpoint):
        staging = self.output / '.checkpoint-uploads' / checkpoint.name
        staging.mkdir(parents=True, exist_ok=True)
        archive = staging / (checkpoint.name + '.tar')
        partial = archive.with_suffix('.tar.partial')
        with tarfile.open(partial, 'w:') as tar:
            tar.add(checkpoint, arcname=checkpoint.name)
        os.replace(partial, archive)
        size, sha = archive.stat().st_size, digest(archive)
        destination = f'{self.bucket}/{checkpoint.name}'
        self.api.sync_bucket(str(staging), destination, delete=False)
        bucket, remote = bucket_parts(destination + '/' + archive.name)
        metadata = self.api.get_bucket_file_metadata(bucket, remote)
        if metadata.size != size:
            raise ValueError('Remote size mismatch')
        # Xet hashes are NOT SHA256. Verify actual remote bytes instead.
        with tempfile.TemporaryDirectory(dir=self.output, prefix='.verify-') as tmp:
            downloaded = Path(tmp) / archive.name
            self.api.download_bucket_files(bucket, [(remote, str(downloaded))], raise_on_missing_files=True)
            if downloaded.stat().st_size != size or digest(downloaded) != sha:
                raise ValueError('Remote SHA256 mismatch')
        self.verified.add(checkpoint)
        archive.unlink()
        staging.rmdir()
        newest = max((p for p in self.output.glob('checkpoint-*') if p.is_dir()),
                     key=lambda p: int(p.name.split('-')[-1]))
        for old in list(self.verified):
            if old != newest and old.exists():
                shutil.rmtree(old)
                self.verified.remove(old)
        print(f'Checkpoint uploaded and size/SHA256 verified: {destination}/{archive.name}', flush=True)

    def close(self):
        self.pool.shutdown(wait=True)
        errors = []
        for future in self.pending:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f'{len(errors)} checkpoint upload(s) failed; local copies retained') from errors[0]
