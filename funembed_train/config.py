from pathlib import Path
import json,os,hashlib
TASKS={'T1':['C','O','B'],'T2':['C-C','O-O','B-B'],'T3':['C-O','O-C']}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def load(p,paths=None):
 p=Path(p).resolve();c=read(p)
 if paths:c['paths'].update(read(paths))
 for k,v in c['paths'].items():
  v=os.path.expandvars(v)
  if '$' in v:raise ValueError('set path/environment variable: '+k+'='+v)
  c['paths'][k]=str((p.parent/v).resolve())
 if not c['tasks'] or not set(c['tasks'])<=set(TASKS):raise ValueError('unknown/empty tasks')
 for t,w in c['tasks'].items():
  if type(w)!=int or w<1:raise ValueError('positive integer task weights required')
  ds=c['directions'][t]
  if not ds or len(set(ds))!=len(ds) or not set(ds)<=set(TASKS[t]):raise ValueError('invalid directions')
 t=c['training'];assert t['world_size']*t['micro_batch']*t['grad_accum']==t['effective_query_batch']
 if t['steps'] is not None and (type(t['steps'])!=int or t['steps']<1):raise ValueError('invalid steps')
 if t['world_size'] not in (1,2,4):raise ValueError('only explicitly configured 1/2/4 ranks; no arbitrary-card claims')
 c['_config_source']=str(p);return c
def work(c):return Path(c['paths']['output_root'])/c['experiment_id']
def env(c,gpus=None):
 e=os.environ.copy();w=work(c);e.update(FUNEMBED_WORK=str(w),FUNEMBED_CLASSES=str(w/'inputs/classes.json'),FUNEMBED_CLASSIFICATION_WORK=str(w/'classification'),SWIFT_ROOT=c['paths']['swift_root'],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1')
 package=str(Path(__file__).resolve().parent.parent);e['PYTHONPATH']=os.pathsep.join([package,c['paths']['swift_root'],e.get('PYTHONPATH','')])
 if gpus is not None:e['CUDA_VISIBLE_DEVICES']=gpus
 return e
