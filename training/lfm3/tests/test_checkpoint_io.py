import tempfile, pathlib, shutil, tarfile
from types import SimpleNamespace
from checkpoint_io import CheckpointUploads, restore
class API:
 def __init__(self, root): self.root=pathlib.Path(root)
 def sync_bucket(self, src, dst, **kw):
  target=self.root/dst.removeprefix('hf://buckets/')
  target.mkdir(parents=True,exist_ok=True)
  for p in pathlib.Path(src).iterdir(): shutil.copy(p,target/p.name)
 def get_bucket_file_metadata(self,b,p): return SimpleNamespace(size=(self.root/b/p).stat().st_size)
 def download_bucket_files(self,b,files,**kw):
  for remote,local in files: shutil.copy(self.root/b/remote,local)
with tempfile.TemporaryDirectory() as tmp:
 root=pathlib.Path(tmp); out=root/'out';out.mkdir(); api=API(root/'remote')
 def checkpoint(n):
  p=out/f'checkpoint-{n}';p.mkdir()
  for name in ('trainer_state.json','optimizer.pt','scheduler.pt','training-policy.json'): (p/name).write_text('{}')
  return p
 a,b=checkpoint(100),checkpoint(200)
 u=CheckpointUploads(out,'hf://buckets/user/bucket/run',api);u.submit(a);u.submit(b);u.close()
 assert not a.exists() and b.exists()
 assert not list(out.rglob('*.tar'))
 restored=restore('hf://buckets/user/bucket/run/checkpoint-100/checkpoint-100.tar',root/'restored',api)
 assert pathlib.Path(restored).is_dir()
 assert restore('latest',out,api)==str(b)
 print('PASS serialized upload, remote size+SHA256 readback, retention, bucket tar restore, latest resume')
