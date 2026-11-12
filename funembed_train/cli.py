import argparse,copy,json,sys,os,subprocess,hashlib,importlib.metadata
from pathlib import Path
from .config import *
from . import workflow as wf

def doctor(c):
 fw=Path(c['paths']['swift_root']);lock=read(Path(__file__).with_name('FRAMEWORK_LOCK.json'));bad=[]
 for n,h in lock['source_sha256'].items():
  if not (fw/n).is_file() or sha(fw/n)!=h:bad.append(n)
 if bad:raise ValueError('ms-swift source differs from tested lock: '+str(bad[:8]))
 model=Path(c['paths']['model_root'])
 for n in ['config.json','tokenizer_config.json','preprocessor_config.json','scripts/qwen3_vl_embedding.py']:
  if not (model/n).is_file():raise FileNotFoundError(model/n)
 index=model/'model.safetensors.index.json'
 if index.exists():
  for shard in set(read(index)['weight_map'].values()):
   if not (model/shard).is_file():raise FileNotFoundError(model/shard)
 elif not (model/'model.safetensors').is_file():raise FileNotFoundError('model safetensors/index missing')
 report={'python':sys.version,'minimum_python':'3.13','packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','numpy','qwen-vl-utils','scikit-learn']},'framework_commit':lock['base_commit'],'framework_files_checked':len(lock['source_sha256']),'CUDA_loaded':False}
 report['model_metadata_sha256']={n:sha(model/n) for n in ['config.json','tokenizer_config.json','preprocessor_config.json','model.safetensors.index.json'] if (model/n).exists()}
 report['code_commit']=Path(__file__).with_name('revision.txt').read_text().strip()
 if report['code_commit'].startswith('$'):
  result=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,capture_output=True,text=True)
  report['code_commit']=result.stdout.strip() if result.returncode==0 else 'uncommitted; see source_fingerprints.json'
 package=Path(__file__).parent.parent
 fingerprint={str(p.relative_to(package)):sha(p) for directory in ['funembed_train','funembed_eval'] for p in (package/directory).rglob('*.py')}
 write(work(c)/'source_fingerprints.json',fingerprint)
 write(work(c)/'environment.json',report);return report

def main():
 p=argparse.ArgumentParser(description='FunEmbed configurable training and evaluation; no implicit GPU actions')
 p.add_argument('action',choices=['check','prepare','train','resume','evaluate','select','lp','run','export','synthetic-demo','retry-stage','verify-export']);p.add_argument('--stage');p.add_argument('--config');p.add_argument('--paths');p.add_argument('--gpus',default='0,1');p.add_argument('--smoke',action='store_true');p.add_argument('--checkpoint',type=int);p.add_argument('--split',choices=['val','internal_test','external_test'],default='val');p.add_argument('--modality',choices=list('COB'),default='B');p.add_argument('--environment',action='store_true');p.add_argument('--output');p.add_argument('--include-queries',action='store_true');p.add_argument('--include-predictions',action='store_true');p.add_argument('--include-embeddings',action='store_true');p.add_argument('--include-weights',action='store_true');a=p.parse_args()
 if a.action=='verify-export':
  if not a.output:p.error('--output is required')
  from .export import validate_archive
  print(json.dumps({k:v for k,v in validate_archive(a.output).items() if k!='files'},indent=2));return
 if a.action=='synthetic-demo':
  from .synthetic import demo
  print(json.dumps(demo(Path(a.output or 'synthetic_output')),indent=2));return
 if not a.config:p.error('--config is required')
 c=load(a.config,a.paths)
 if a.smoke:
  if c['selection']['mode']!='within_split':c['selection']['mode']='within_split';c['evaluation']['case_protocol']='within_split'
  cycle=sum(c['tasks'].values())
  import math
  cycles=math.lcm(*(len(c['directions'][t])//math.gcd(len(c['directions'][t]),w) for t,w in c['tasks'].items()))
  cycle*=cycles;c['training'].update(steps=cycle*2,save_steps=cycle);c['candidate_steps']=[cycle,cycle*2];c['experiment_id']+='_smoke';c['smoke']=True
 w=work(c);w.mkdir(parents=True,exist_ok=True)
 if a.action=='retry-stage':
  if not a.stage or '/' in a.stage or '..' in a.stage:p.error('--stage requires an exact stage name')
  import time,shutil
  state=w/'stages'/f'{a.stage}.json';old=read(state)
  if old['status']=='completed':raise ValueError('completed stages cannot be reset')
  if old.get('pid') and old.get('exit_code') is None:
   try:os.kill(old['pid'],0)
   except ProcessLookupError:pass
   else:raise RuntimeError('recorded PID still exists; inspect process before retry')
  dest=w/'failed_attempts'/(a.stage+'_'+str(time.time_ns()));dest.mkdir(parents=True)
  shutil.move(str(state),str(dest/state.name));log=w/'logs'/f'{a.stage}.log'
  if log.exists():shutil.move(str(log),str(dest/log.name))
  print('Saved failed attempt in',dest,'; rerun the requested action explicitly.');return
 if a.action=='check':
  from .inputs import prepare
  print(json.dumps(prepare(c),indent=2))
  if a.environment:print(json.dumps(doctor(c),indent=2))
  return
 if a.action=='prepare':wf.prepare(c);return
 if a.action in ['run','train','resume']:doctor(c)
 if a.action=='run':wf.full(c,a.gpus);return
 if a.action=='train':wf.prepare(c);wf.train(c,a.gpus);return
 if a.action=='resume':
  if a.checkpoint is None:p.error('--checkpoint required')
  wf.train(c,a.gpus,resume=w/'train'/f'checkpoint-{a.checkpoint}')
  final=w/'train'/f'checkpoint-{c["training"]["steps"]}'/'trainer_state.json'
  if final.exists() and read(final)['global_step']==c['training']['steps']:
   s=read(w/'stages/train.json');s.update(status='completed',completed_by_resume_from=a.checkpoint);write(w/'stages/train.json',s)
  return
 if a.action=='evaluate':
  if a.checkpoint is None:p.error('--checkpoint required; 0 is original model')
  wf.evaluation(c,a.checkpoint,a.split,a.gpus.split(',')[0]);return
 if a.action=='select':
  steps=wf.candidate_steps(c);wf.select(c,steps,[w/'evaluation'/f'checkpoint-{s}'/'val/results/summary.json' for s in steps]);return
 if a.action=='lp':
  if a.checkpoint is None:p.error('--checkpoint required')
  wf.lp(c,a.checkpoint,a.modality,a.gpus.split(',')[0]);return
 if a.action=='export':
  from .export import export_results
  print(json.dumps(export_results(w,a.output or w/'result_export_manual.tar.gz',a.include_queries,a.include_predictions,a.include_embeddings,a.include_weights),indent=2))
if __name__=='__main__':main()
