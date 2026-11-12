"""Strict val->val selection for new experiments. No model import or test access."""
from pathlib import Path
import json,hashlib,copy,math,argparse,re,sys

from funembed_eval.data import read_manifest,label_ids
from funembed_eval.selection import role_rows

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
TASKS=('C_C','O_O','B_B')
def inspect(path):
 p=Path(path);s=read(p);c=s['config'];e=c['encoder']
 for d in (s,c):
  if d['case_protocol']!='within_split' or d['query_split']!='val' or d['gallery_split']!='val':raise ValueError('only within_split val->val accepted')
 if c.get('smoke_query_limit') or c.get('smoke_gallery_limit'):raise ValueError('smoke results cannot select formal checkpoints')
 done=read(p.parent/'complete.json')
 if done['status']!='completed' or done['summary_sha256']!=sha(p):raise ValueError('incomplete or changed result')
 cache=Path(c['cache']);fp=read(cache/'fingerprint.json');cm=read(cache/'complete.json')
 if stable(fp)!=s['cache_fingerprint'] or cm['fingerprint']!=stable(fp):raise ValueError('cache fingerprint differs')
 if fp['case_protocol']!='within_split' or fp['splits']!=['val','val']:raise ValueError('cache split differs')
 for key in ('encoder','instruction','aggregation','cohort','case_directions','evaluate_text','text_kind'):
  if fp[key]!=c[key]:raise ValueError('config/cache mismatch: '+key)
 if fp['instruction']!='I1' or fp['aggregation']!='joint':raise ValueError('expected I1 joint')
 if e['image_max_tokens']!=1024 or e['max_length']!=8192:raise ValueError('input budget differs')
 if not fp['roles']['query'] or fp['roles']['query']!=fp['roles']['gallery']:raise ValueError('query/gallery role IDs differ')
 for role in ('query','gallery'):
  if len(fp['roles'][role])!=len(set(fp['roles'][role])):raise ValueError('duplicate role IDs')
 for key in ('manifest','classes','texts'):
  if sha(c[key])!=fp[key+'_sha256']:raise ValueError('changed data/text mapping: '+key)
 rows=read_manifest(c['manifest'],c['image_root']);roles=role_rows(c,rows)
 for role in ('query','gallery'):
  if [r['case_id'] for r in roles[role]]!=fp['roles'][role]:raise ValueError('actual role set differs')
 if (c.get('query_ids') or c.get('gallery_ids')) and 'short_fixed_subset' not in c['protocol_id'] and 'synthetic' not in c['protocol_id']:raise ValueError('formal selection requires complete val cohort')
 # Verify all encoded artifacts; no model/NPZ loading required.
 for name,h in cm['artifacts'].items():
  if sha(cache/name)!=h:raise ValueError('changed artifact '+name)
 for name in ('val_C.npz','val_O.npz','val_B.npz'):
  if name not in cm['artifacts'] or not fp['encoding_groups'].get(name[:-4]):raise ValueError('missing modality cache')
 # Explicit role subsets are allowed only when identical across candidates; use a distinct protocol ID for short-test selection.
 adapter=e.get('adapter');match=re.fullmatch(r'checkpoint-(\d+)',Path(adapter or '').name)
 if not match:raise ValueError('adapter must identify a checkpoint-N')
 if not fp.get('adapter_sha256'):raise ValueError('missing adapter fingerprint')
 expected_tasks={x.replace('-','_') for x in c['case_directions']}
 if c.get('evaluate_text'):expected_tasks.update(m+'_text_'+c['text_kind'] for m in c['text_modalities'])
 if set(s['protocols'])!=set(c['protocols']):raise ValueError('incomplete label protocols')
 for protocol in c['protocols']:
  if set(s['protocols'][protocol]['tasks'])!=expected_tasks:raise ValueError('incomplete configured tasks')
 step=int(match.group(1));r=s['protocols']['modal'];scores={}
 for t in TASKS:
  v=r['tasks'][t];n=v['n_queries_valid'];score=v['ndcg@5']
  if not n or not math.isfinite(score) or not 0<=score<=1:raise ValueError('invalid score/coverage')
  if v['n_queries_label_valid']!=v['n_queries_valid']+v['n_queries_no_relevant']:raise ValueError('inconsistent coverage')
  if v['n_queries_available']!=r['query_counts'][t[0]]['available']:raise ValueError('inconsistent available count')
  qfile=p.parent/'modal'/f'per_query__{t}.jsonl'
  records=[json.loads(x)['metrics'] for x in qfile.read_text(encoding='utf-8').splitlines() if x.strip()]
  expected=[r['tie_key'] for r in roles['query'] if label_ids(r,t[0],'modal')]
  if [x['sample_path'] for x in records]!=expected:raise ValueError('actual query IDs differ')
  if r['gallery_counts'][t[-1]]!=sum(bool(label_ids(x,t[-1],'modal')) for x in roles['gallery']):raise ValueError('gallery count differs')
  vals=[x['ndcg@5'] for x in records if not x['no_relevant']]
  if len(vals)!=n or len(records)!=v['n_queries_label_valid'] or abs(sum(vals)/n-score)>1e-12:raise ValueError('per-query aggregate mismatch')
  scores[t]=score
 identity=copy.deepcopy(fp);identity.pop('adapter_sha256',None);identity['encoder'].pop('adapter',None)
 # Runtime device selection does not change the mathematical comparison; every other encoder condition remains fixed.
 identity['encoder'].pop('device',None)
 contract={'protocol_id':s['protocol_id'],'protocols':c['protocols'],'fingerprint_without_adapter':identity,'query_counts':r['query_counts'],'gallery_counts':r['gallery_counts'],'coverage':{t:{k:v for k,v in r['tasks'][t].items() if k.startswith(('n_queries','n_gallery','exclusions','coverage'))} for t in TASKS}}
 return {'step':step,'checkpoint':adapter,'directions':scores,'score':sum(scores.values())/3,'result':str(p.resolve()),'sha256':sha(p),'diagnostics':{k:v for k,v in r['tasks'].items() if k not in TASKS}},contract

def select(paths):
 if not paths:raise ValueError('at least one completed candidate required')
 entries=[];reference=None
 for p in paths:
  item,contract=inspect(p)
  if reference is None:reference=contract
  elif reference!=contract:raise ValueError('mixed comparison conditions/data/roles/coverage')
  entries.append(item)
 if len({x['step'] for x in entries})!=len(entries):raise ValueError('duplicate step')
 entries.sort(key=lambda x:x['step']);winner=max(entries,key=lambda x:(x['score'],-x['step']))
 return {'protocol_id':reference['protocol_id'],'metric':'modal val->val equal mean C_C/O_O/B_B NDCG@5','tie':'earlier optimizer step (exact score equality)','test_sets_accessed':False,'comparison_contract_sha256':stable(reference),'candidates':entries,'selected':winner}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summaries',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=select(a.summaries)
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(d['selected'],ensure_ascii=False))
if __name__=='__main__':main()
