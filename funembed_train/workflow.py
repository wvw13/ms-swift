from pathlib import Path
import os,sys,subprocess,time,signal,json,copy,math,hashlib
from .config import *
from .inputs import prepare as inputs
V=Path(__file__).parent/'vendor/v1'
def run(c,name,args,gpus=None,extra=None):
 w=work(c);w.mkdir(parents=True,exist_ok=True);status=w/'stages'/f'{name}.json';command=[str(x) for x in args];e=env(c,gpus);e.update(extra or {});signature=hashlib.sha256(json.dumps({'command':command,'extra':extra,'gpus':gpus},sort_keys=True).encode()).hexdigest()
 if status.exists():
  s=read(status)
  if s.get('signature')!=signature:raise ValueError('stage command changed: '+name)
  if s['status']=='completed':return
  raise RuntimeError('stage incomplete/failed: inspect and use explicit resume, no automatic retry: '+name)
 hardware=None
 if gpus:
  try:
   hardware=subprocess.check_output(['nvidia-smi','-i',gpus,'--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv'],text=True,timeout=5).strip()
  except (OSError,subprocess.SubprocessError):pass
 log=w/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True);start=time.monotonic()
 with log.open('x',encoding='utf-8') as f:
  p=subprocess.Popen(command,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(status,dict(status='running',pid=p.pid,signature=signature,command=command,log=str(log),gpus=gpus));write(w/'status.json',dict(status='running',stage=name,pid=p.pid,log=str(log)))
  try:code=p.wait(timeout=c.get('stage_timeout_hours',96)*3600)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=-9
 result=dict(status='completed' if code==0 else 'failed',signature=signature,command=command,log=str(log),exit_code=code,seconds=time.monotonic()-start,gpus=gpus,hardware=hardware);write(status,result)
 if code:write(w/'status.json',dict(status='failed',stage=name,log=str(log)));raise RuntimeError(name+' failed: '+str(log))
def prepare(c):
 inputs(c);w=work(c);resolved=w/'resolved.json'
 if resolved.exists() and read(resolved)!=c:raise ValueError('resolved config changed, use new experiment_id')
 write(resolved,c);t=c['training']
 if t['steps'] is None:raise ValueError('set an explicit approved steps budget before materializing samples')
 if (w/'pairs').exists():
  run(c,'verify_pairs',[sys.executable,V/'verify.py',w/'pairs']);return
 args=[sys.executable,V/'prepare.py','--base',w/'inputs/phase2','--output',w/'pairs','--experiment','CUSTOM','--cfp-definitions',w/'inputs/definition_C.json','--oct-definitions',w/'inputs/definition_O.json','--text-kind','definition','--steps',str(t['steps']),'--world-size',str(t['world_size']),'--micro-batch',str(t['micro_batch']),'--grad-accum',str(t['grad_accum']),'--normal-fraction',str(t['normal_fraction']),'--seed',str(t['seed']),'--negatives',str(t['K']),'--task-weights',json.dumps(c['tasks']),'--directions-json',json.dumps(c['directions'])]
 run(c,'prepare_pairs',args);run(c,'verify_pairs',[sys.executable,V/'verify.py',w/'pairs'])
def train(c,gpus,resume=None,stop=None):
 t=c['training'];w=work(c)
 if len(gpus.split(','))!=t['world_size']:raise ValueError('GPU count differs from frozen schedule')
 args=[sys.executable,V/'train.py','--pairs',w/'pairs','--output',w/'train','--model',c['paths']['model_root'],'--gpus',gpus,'--image-max-tokens','1024','--max-length','8192','--learning-rate',str(t['lr']),'--lora-rank',str(t.get('lora_rank',16)),'--lora-alpha',str(t.get('lora_alpha',32)),'--lora-dropout',str(t.get('lora_dropout',.05)),'--temperature',str(t.get('temperature',.1)),'--attn-impl','sdpa','--save-steps',str(t['save_steps']),'--save-total-limit','0','--warmup-ratio',str(t.get('warmup_ratio',.05)),'--weight-decay',str(t.get('weight_decay',.01)),'--execute']
 extra={}
 if c.get('smoke'):extra['FUNEMBED_E2E_AUDIT']='1'
 if stop:extra['FUNEMBED_VALIDATION_STOP_AFTER']=str(stop)
 if resume:args+=['--resume',str(resume)]
 name='train'+('_stop'+str(stop) if stop else '')+('_resume'+Path(resume).name if resume else '')
 run(c,name,args,gpus,extra)
def evaluation(c,step,split,gpu,adapter=None):
 w=work(c);tag='original' if step==0 else f'checkpoint-{step}';d=w/'evaluation'/tag/split;protocol=c['evaluation'].get('case_protocol','within_split');cfg=dict(manifest=str(w/'inputs/manifest.jsonl'),classes=str(w/'inputs/classes.json'),texts=str(w/'inputs/texts.json'),image_root=c['paths']['data_root'],cache=str(d/'cache'),output=str(d/'results'),query_split=split,gallery_split=split if protocol=='within_split' else 'internal_train',case_protocol=protocol,protocol_id='linyan_I1_joint_'+protocol+('_short_fixed_subset' if c.get('smoke') else '_batch1'),instruction='I1',aggregation='joint',cohort='input_available',protocols=['modal','combined'],case_directions=c['evaluation']['case_directions'],text_modalities=list('COB'),text_kind='definition',evaluate_text=True,encoder=dict(backend='swift',model=c['paths']['model_root'],adapter=str(adapter or w/'train'/tag) if step else None,framework_path=c['paths']['swift_root'],model_type='qwen3_vl_emb',template_type='qwen3_vl_emb',representation='swift_embedding',batch_size=1,image_max_tokens=1024,max_length=8192,device='cuda:0'))
 if c.get('smoke'):
  ids=[json.loads(l)['case_id'] for l in (w/'inputs/manifest.jsonl').read_text().splitlines() if json.loads(l)['split']==split][:16];cfg['query_ids']=ids;cfg['gallery_ids']=ids
 path=d/'config.json';write(path,cfg)
 for action in ['check','encode','evaluate']:
  args=[sys.executable,'-m','funembed_eval',action,'--config',path]
  if action=='check':args+=['--files','--environment']
  run(c,f'{tag}_{split}_{action}',args,gpu)
 run(c,f'{tag}_{split}_audit',[sys.executable,'-m','funembed_eval','audit-results','--output',d/'results'])
 return d/'results/summary.json'
def lp(c,step,m,gpu,stage='all',tiny=False):
 w=work(c);args=[sys.executable,'-m','funembed_train.lp_worker','--resolved',w/'resolved.json','--modality',m,'--output',w/'lp'/f'checkpoint-{step}'/m,'--stage',stage]
 if step:args+=['--adapter',w/'train'/f'checkpoint-{step}']
 if tiny:args+=['--tiny']
 run(c,f'LP_{step}_{m}_{stage}',args,gpu)
 from .lp_summary import summarize
 summarize(w/'lp'/f'checkpoint-{step}'/m)
def select(c,steps,summaries):
 w=work(c);mode=c['selection']['mode']
 if mode=='within_split':
  from .select_within import select as selector
  result=selector(summaries)
 elif mode=='reference_train':
  entries=[]
  for step,p in zip(steps,summaries):
   d=read(p);assert d['case_protocol']=='reference_train' and d['query_split']=='val';tasks=d['protocols']['modal']['tasks'];score=sum(tasks[t]['ndcg@5'] for t in ['C_C','O_O','B_B'])/3;entries.append(dict(step=step,score=score,checkpoint=str(w/'train'/f'checkpoint-{step}')))
  result=dict(candidates=entries,selected=max(entries,key=lambda x:(x['score'],-x['step'])),mode=mode)
 elif mode=='historical_B_LP':
  entries=[]
  for step in steps:
   vals=[read(w/'lp'/f'checkpoint-{step}'/'B'/f'seed{s}/metrics/val.json')['mAP'] for s in range(3)];entries.append(dict(step=step,score=sum(vals)/3,checkpoint=str(w/'train'/f'checkpoint-{step}')))
  result=dict(candidates=entries,selected=max(entries,key=lambda x:(x['score'],-x['step'])),mode=mode)
 else:raise ValueError(mode)
 if [x['step'] for x in result['candidates']]!=steps:raise ValueError('incomplete candidate list')
 write(w/'selected.json',result);return result['selected']['step']
def full(c,gpus):
 import fcntl
 w=work(c);w.mkdir(parents=True,exist_ok=True);lock=(w/'.workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

 if (w/'result_export.tar.gz').exists() and (w/'status.json').exists() and read(w/'status.json').get('status')=='completed':return
 if c.get('baseline'):inputs(c);write(w/'resolved.json',c)
 else:prepare(c)
 gpu=gpus.split(',')[0];t=c['training']
 if c.get('baseline'):
  for split in ['val','internal_test','external_test']:evaluation(c,0,split,gpu)
  for m in c.get('lp_modalities',list('COB')):lp(c,0,m,gpu)
 else:
  if c.get('smoke'):
   train(c,gpus,stop=t['save_steps']);train(c,gpus,resume=w/'train'/f'checkpoint-{t["save_steps"]}')
   from .acceptance import check_training
   os.environ.update(env(c));check_training(c)
  else:train(c,gpus)
  steps=c.get('candidate_steps') or list(range(t['save_steps'],t['steps']+1,t['save_steps']))
  summaries=[]
  for step in steps:
   summaries.append(evaluation(c,step,'val',gpu))
   if c['selection']['mode']=='historical_B_LP':lp(c,step,'B',gpu,'val')
  step=select(c,steps,summaries)
  if c.get('smoke'):lp(c,step,'C',gpu,'val',tiny=True)
  else:
   for split in ['internal_test','external_test']:evaluation(c,step,split,gpu)
   for m in c.get('lp_modalities',list('COB')):lp(c,step,m,gpu)
 write(w/'status.json',{'status':'completed','smoke':bool(c.get('smoke'))})
 from .export import export_results
 export_results(w,w/'result_export.tar.gz')
