"""Aggregate completed classification heads, keeping encoder/head seeds distinct."""
from pathlib import Path
import csv,statistics
from .config import read,write
METRICS={'mAP':'mAP','AUC':'macro_auc','micro-F1':'micro_f1','macro-F1':'macro_f1'}
def summarize(root):
 root=Path(root);seeds=[];groups={}
 for folder in sorted(root.glob('seed*')):
  seed=int(folder.name[4:]);train=read(folder/'train_complete.json')
  logs=[read(p) for p in sorted((folder/'logs').glob('epoch_*.json'))]
  info={'head_seed':seed,'stopped_early':train['stopped_early'],'actual_epochs':max((x['epoch'] for x in logs),default=None)}
  if logs:info['best_epoch']=max(logs,key=lambda x:(x['val_mAP'],-x['epoch']))['epoch']
  if (folder/'complete.json').exists():info.update({k:v for k,v in read(folder/'complete.json').items() if k!='signature'})
  scores={}
  for p in (folder/'metrics').glob('*.json'):
   d=read(p);scores[p.stem]={name:d[key] for name,key in METRICS.items()};scores[p.stem]['threshold']=d.get('threshold');groups.setdefault(p.stem,[]).append(scores[p.stem])
  seeds.append(info|{'metrics':scores})
 aggregate={s:{name:{'mean':statistics.mean(v[name] for v in vals),'sample_std':statistics.stdev(v[name] for v in vals) if len(vals)>1 else None,'n_heads':len(vals)} for name in METRICS} for s,vals in groups.items()}
 result={'encoder_training_seed':0,'uncertainty_unit':'classification_head_seed','heads':seeds,'aggregate':aggregate};write(root/'summary.json',result)
 with (root/'summary.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['split','metric','mean','sample_std','n_heads'])
  for s,metrics in aggregate.items():
   for k,v in metrics.items():w.writerow([s,k,v['mean'],v['sample_std'],v['n_heads']])
 return result
