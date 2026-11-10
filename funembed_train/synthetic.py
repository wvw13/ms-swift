"""Artificial identifiers, labels and image bytes for CPU-only contract tests."""
from .config import *
import hashlib

def fixture(d):
 d=Path(d).resolve();private=d/'private';images=d/'images';private.mkdir(parents=True);images.mkdir();rows=[]
 classes=[{'id':i,'label':f'synthetic_class_{i}','text':f'SYNTHETIC name {i}'} for i in range(22)];texts=[]
 for i in range(22):
  cc=f'SYNTHETIC C description {i}';oo=f'SYNTHETIC O description {i}';texts.append({'id':i,'name':classes[i]['text'],'definition':{'C':cc,'O':oo,'B':f'CFP features: {cc} OCT features: {oo}'}})
 for split,n in [('internal_train',32),('val',16),('internal_test',16),('external_test',16)]:
  for i in range(n):
   cid=f'synthetic_{split}_{i}';ls=[[21],[0],[1],[0,1]][i%4];paths=[cid+'_cfp.jpg']+[cid+f'_{j:03}.jpg' for j in range(7,13)]
   for p in paths:(images/p).write_bytes(('SYNTHETIC CPU FIXTURE '+p).encode())
   rows.append(dict(case_id=cid,tie_key=cid,patient_id=cid,center='synthetic',split=split,input_ids={m:cid+'_'+m for m in 'COB'},images={'cfp':paths[0],'oct':[{'slice':j,'path':paths[j-6]} for j in range(7,13)]},labels={g:{'ids':ls,'state':'normal' if ls==[21] else 'disease'} for g in ['cfp','oct','combined']},image_md5=[hashlib.md5((images/p).read_bytes()).hexdigest() for p in paths]))
 (private/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');write(private/'classes.json',classes);write(private/'texts.json',texts)
 c=dict(experiment_id='synthetic_e4',paths=dict(model_root=str(d/'fake_model'),data_root=str(images),private_input_root=str(private),swift_root=str(d/'fake_framework'),output_root=str(d/'outputs')),tasks={'T1':3,'T2':3,'T3':2},directions={k:list(v) for k,v in TASKS.items()},training=dict(steps=16,save_steps=8,world_size=1,micro_batch=1,grad_accum=1,effective_query_batch=1,K=4,normal_fraction=.2,seed=0,lr=5e-5),selection={'mode':'within_split'},evaluation={'case_protocol':'within_split','case_directions':['C-C','O-O','B-B','O-C','C-O']},lp_modalities=list('COB'))
 write(d/'config.json',c);return c

def demo(d):
 from . import workflow as wf
 from funembed_eval.config import load as evload
 from funembed_eval.data import read_manifest
 from funembed_eval.encode import execute as encode
 from funembed_eval.evaluate import execute,audit
 from .export import export_results
 c=fixture(d);wf.prepare(c);w=work(c)
 # Use same eval protocol with an explicitly synthetic vector backend.
 cfg=dict(manifest=str(w/'inputs/manifest.jsonl'),classes=str(w/'inputs/classes.json'),texts=str(w/'inputs/texts.json'),image_root=c['paths']['data_root'],cache=str(w/'evaluation/original/val/cache'),output=str(w/'evaluation/original/val/results'),query_split='val',gallery_split='val',case_protocol='within_split',protocol_id='synthetic_within_split',instruction='I1',aggregation='joint',cohort='input_available',protocols=['modal','combined'],case_directions=c['evaluation']['case_directions'],text_kind='definition',evaluate_text=True,text_modalities=list('COB'),encoder={'backend':'synthetic','model':None,'adapter':None,'image_max_tokens':1024,'max_length':8192,'batch_size':1})
 p=w/'cpu_eval.json';write(p,cfg);cfg=evload(p);rows=read_manifest(cfg['manifest'],cfg['image_root']);encode(cfg,rows);execute(cfg,rows);report=audit(cfg['output']);package=export_results(w,w/'cpu_results.tar.gz');return {'sampling':'all eight directions materialized/verified','evaluation_audit':report,'export':package,'GPU_used':False}
