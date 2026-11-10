"""Isolated process: shared historical classification encoder and LP engine."""
from pathlib import Path
import argparse,json,sys,os
from types import SimpleNamespace
from .config import read,write,env,work

def main():
 p=argparse.ArgumentParser();p.add_argument('--resolved',required=True);p.add_argument('--adapter');p.add_argument('--modality',required=True,choices=list('COB'));p.add_argument('--output',required=True);p.add_argument('--stage',choices=['val','test','all'],default='all');p.add_argument('--tiny',action='store_true');a=p.parse_args();c=read(a.resolved);os.environ.update(env(c));v=Path(__file__).parent/'vendor/classification';sys.path.insert(0,str(v))
 import torch,numpy as np
 import run as engine
 from peft import PeftModel,get_peft_model_state_dict
 from safetensors.torch import load_file
 torch.set_num_threads(4);cfg=engine.load_config(str(work(c)/'classification/configs/common.json'));out=Path(a.output);cache=out/'cache';prot=engine.protocol(cfg,'2b',a.modality)
 if a.tiny:cfg['lp']['epochs']=2;cfg['lp']['patience']=1
 if a.adapter:
  adapter=Path(a.adapter);prot['encoder_adapter']={'path':str(adapter),'sha256':{n:engine.digest(adapter/n) for n in ['adapter_config.json','adapter_model.safetensors']}}
  original=engine.Classifier
  class Adapted(original):
   def __init__(self,cfg,size,method,device):
    super().__init__(cfg,size,'lp',device);self.encoder=PeftModel.from_pretrained(self.encoder,str(adapter),is_trainable=False);self.encoder.requires_grad_(False);self.eval()
    saved=load_file(str(adapter/'adapter_model.safetensors'));actual=get_peft_model_state_dict(self.encoder)
    assert set(saved)==set(actual) and all(torch.equal(actual[k].detach().cpu(),saved[k].to(actual[k].dtype)) for k in saved)
    write(cache/'adapter_check.json',{'tensors':len(saved),'frozen':True})
  engine.Classifier=Adapted
 engine.cache_dir=lambda p:cache;device,rank,world=engine.runtime();assert world==1
 args=SimpleNamespace(model='2b',modality=a.modality,method='lp',seed=0,resume=True,run=None)
 if a.tiny:
  original_rows=engine.rows_for
  engine.rows_for=lambda split:original_rows(split)[:32 if split=='internal_train' else 16]
 if a.stage in ['val','all']:
  engine.SPLITS=('internal_train','val');engine.extract(args,cfg,device,rank,world,prot)
  for seed in ([0] if a.tiny else [0,1,2]):
   args.seed=seed;args.run=str(out/f'seed{seed}');engine.train(args,cfg,device,rank,world,prot)
 if a.stage in ['test','all']:
  engine.SPLITS=('internal_test','external_test');engine.extract(args,cfg,device,rank,world,prot)
  for seed in ([0] if a.tiny else [0,1,2]):
   args.seed=seed;args.run=str(out/f'seed{seed}');engine.evaluate(args,cfg,device,rank,world,prot)
 write(out/f'{a.stage}_complete.json',{'status':'completed','modality':a.modality,'seeds':[0] if a.tiny else [0,1,2],'tiny':a.tiny,'stage':a.stage})
if __name__=='__main__':main()
