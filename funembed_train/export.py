"""Allowlisted aggregate export; detailed artifacts need separate explicit options."""
from pathlib import Path
import json,csv,tarfile,tempfile,shutil,hashlib,os,gzip,time,fcntl
from contextlib import contextmanager
from .config import read,write,sha

class InvalidArchive(ValueError):
 pass

def validate_archive(output):
 """Read gzip through EOF, reject links/traversal, extract safely and verify every file."""
 output=Path(output)
 try:
  with gzip.open(output,'rb') as stream:
   while stream.read(2**20):pass
  with tempfile.TemporaryDirectory() as temp:
   dest=Path(temp)
   with tarfile.open(output,'r:gz') as tar:
    members=tar.getmembers();names=[m.name for m in members]
    if len(names)!=len(set(names)):raise ValueError('duplicate archive members')
    for m in members:
     path=Path(m.name)
     if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='funembed_results':raise ValueError('unsafe archive path')
     if not (m.isfile() or m.isdir()):raise ValueError('links/special members prohibited')
    tar.extractall(dest,filter='data')
   base=dest/'funembed_results';manifest=read(base/'FILES.sha256.json')
   actual={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file() and p.name!='FILES.sha256.json'}
   if not manifest or manifest!=actual:raise ValueError('file inventory/hash mismatch')
   if not {'resolved_config.json','export_policy.json'}<=actual.keys():raise ValueError('required export files missing')
   return {'file_count':len(actual)+1,'bytes':output.stat().st_size,'sha256':sha(output),'archive':str(output),'files':actual}
 except (OSError,EOFError,tarfile.TarError,ValueError,KeyError) as error:
  raise InvalidArchive(f'Invalid result archive {output}: {error}') from error

@contextmanager
def output_lock(output):
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 with output.with_name(output.name+'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  yield

def export_results(root,output,queries=False,predictions=False,embeddings=False,weights=False,*,recover_invalid=False):
 with output_lock(output):
  return _export_results(root,output,queries,predictions,embeddings,weights,recover_invalid=recover_invalid)

def _export_results(root,output,queries=False,predictions=False,embeddings=False,weights=False,*,recover_invalid=False):
 root=Path(root).resolve();output=Path(output).resolve()
 with tempfile.TemporaryDirectory() as temp:
  dest=Path(temp)/'funembed_results';dest.mkdir()
  cfg=read(root/'resolved.json');cfg['paths']={k:'<'+k+'>' for k in cfg['paths']};cfg.pop('_config_source',None);write(dest/'resolved_config.json',cfg)
  for n in ['source_fingerprints.json','environment.json','resume_acceptance.json']:
   if (root/n).exists():write(dest/n,read(root/n))
  if (root/'inputs/complete.json').exists():
   d=read(root/'inputs/complete.json');write(dest/'data_fingerprint.json',{k:d[k] for k in ['source_sha256','counts','classification_counts']})
  for p in root.glob('evaluation/*/*/results/summary.json'):
   d=read(p);safe={k:d[k] for k in ['protocol_id','case_protocol','query_split','gallery_split','protocols']};write(dest/p.relative_to(root),safe)
  if (root/'selected.json').exists():
   d=read(root/'selected.json');write(dest/'selected.json',{'metric':d.get('metric',d.get('mode')),'selected_step':d['selected']['step'],'candidates':[{k:x[k] for k in ['step','score','directions'] if k in x} for x in d['candidates']]})
  for p in root.glob('lp/**/metrics/*.json'):write(dest/p.relative_to(root),read(p))
  for p in root.glob('lp/**/train_complete.json'):write(dest/p.relative_to(root),{k:v for k,v in read(p).items() if k!='signature'})
  for p in root.glob('lp/**/complete.json'):write(dest/p.relative_to(root),{k:v for k,v in read(p).items() if k!='signature'})
  for p in root.glob('lp/**/summary.json'):write(dest/p.relative_to(root),read(p))
  adapters={}
  for cp in sorted((root/'train').glob('checkpoint-*')):
   adapters[cp.name]={p.name:sha(p) for p in cp.glob('adapter*') if p.is_file() and p.suffix in ('.json','.safetensors')}
  write(dest/'model_fingerprints.json',{'base_metadata':read(root/'environment.json').get('model_metadata_sha256',{}) if (root/'environment.json').exists() else {},'checkpoint_adapter_sha256':adapters,'base_weight_content_hash':'not computed; retain model provider/version separately'})
  resources=[]
  for p in root.glob('stages/*.json'):
   d=read(p);resources.append({k:d.get(k) for k in ['status','seconds','gpus','exit_code','hardware'] }|{'stage':p.stem})
  write(dest/'resources.json',resources)
  training_resources={}
  for p in root.glob('train/completed_updates*.jsonl'):
   training_resources[p.name]=[{k:v for k,v in json.loads(line).items() if k in ['step','global_step','rank','task','direction','row_count','step_seconds','elapsed_seconds','num_input_tokens_seen','peak_allocated_GiB','peak_reserved_GiB','rank0_peak_allocated_GiB','rank0_peak_reserved_GiB']} for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
  write(dest/'training_resources.json',training_resources)
  states=sorted(root.glob('train/checkpoint-*/trainer_state.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
  if states:
   history=read(states[-1]).get('log_history',[]);schedule={}
   if (root/'pairs/schedule.jsonl').exists():schedule={x['step']:x for x in map(json.loads,(root/'pairs/schedule.jsonl').read_text().splitlines())}
   with (dest/'loss_lr.csv').open('w',encoding='utf-8') as f:
    cols=['step','task','direction','loss','learning_rate','grad_norm','query_exposures'];w=csv.DictWriter(f,cols);w.writeheader()
    for x in history:
     if 'loss' not in x:continue
     s=schedule.get(x.get('step'),{});w.writerow({k:x.get(k,'') for k in ['step','loss','learning_rate','grad_norm']}|{k:s.get(k,'') for k in ['task','direction']}|{'query_exposures':s.get('row_count','')})
  ready=read(root/'export_ready.json') if (root/'export_ready.json').exists() else {}
  write(dest/'workflow_state.json',{'compute_stages_complete':ready.get('compute_stages_complete',False),'smoke':bool(cfg.get('smoke')),'archive_integrity':'verify with FILES.sha256.json and verify-export'})
  options={'queries':queries,'predictions':predictions,'embeddings':embeddings,'weights':weights};write(dest/'export_policy.json',{'options':options,'default':'aggregate only; no weights, images, patient IDs, vectors or complete stdout'})
  patterns=[]
  if queries:patterns+=['evaluation/**/per_query*.jsonl']
  if predictions:patterns+=['lp/**/predictions/*.npz']
  if embeddings:patterns+=['evaluation/**/cache/*.npz','lp/**/cache/*.npz']
  if weights:patterns+=['train/checkpoint-*/*']
  for pattern in patterns:
   for p in root.glob(pattern):
    if p.is_file():target=dest/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  files={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()};write(dest/'FILES.sha256.json',files)
  output.parent.mkdir(parents=True,exist_ok=True)
  preserved=None
  if output.exists():
   try:existing=validate_archive(output)
   except InvalidArchive:
    if not recover_invalid:raise
    preserved=output.with_name(output.name+f'.invalid.{time.time_ns()}');os.replace(output,preserved)
   else:
    if existing['files']!=files:raise FileExistsError('Existing valid result differs: keep it and choose a new output filename')
    return {k:v for k,v in existing.items() if k!='files'}|{'reused':True}
  fd,name=tempfile.mkstemp(prefix=output.name+'.',suffix='.partial',dir=output.parent);os.close(fd);partial=Path(name)
  # Retain failed partial files for diagnosis. Only a fully verified archive becomes final.
  with tarfile.open(partial,'w:gz') as tar:tar.add(dest,arcname='funembed_results')
  with partial.open('rb') as f:os.fsync(f.fileno())
  checked=validate_archive(partial)
  if checked['files']!=files:raise InvalidArchive('validated inventory differs from intended export')
  os.replace(partial,output)
  directory=os.open(output.parent,os.O_RDONLY)
  try:os.fsync(directory)
  finally:os.close(directory)
  return {k:v for k,v in checked.items() if k!='files'}|{'archive':str(output),'reused':False,'preserved_invalid':str(preserved) if preserved else None}
