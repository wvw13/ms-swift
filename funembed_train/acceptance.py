from .config import *
def check_training(c):
 import sys,torch
 from safetensors.torch import load_file
 sys.path.insert(0,str(Path(__file__).parent/'vendor/v1'))
 from validation_audit import tensor_hash,state_digest
 from .audit_ddp import summarize
 w=work(c);run=w/'train';step=c['training']['save_steps'];final=c['training']['steps'];world=c['training']['world_size'];cp=run/f'checkpoint-{step}'
 summarize(w,w/'pairs',run)
 saved=load_file(str(cp/'adapter_model.safetensors'));opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False);sch=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)
 for rank in range(world):
  begin=read(run/f'validation_begin_step{step}_rank{rank}.json');assert begin['global_step']==step and begin['optimizer']==state_digest(opt) and begin['scheduler']==state_digest(sch)
  weights={k.replace('.default.','.'):v for k,v in begin['lora_sha256'].items()};assert weights.keys()==saved.keys() and all(weights[k]==tensor_hash(v) for k,v in saved.items())
  rng=cp/('rng_state.pth' if world==1 else f'rng_state_{rank}.pth');assert rng.is_file();rs=torch.load(rng,map_location='cpu',weights_only=False);assert {'python','numpy','cpu','cuda'}<=rs.keys()
  end=read(run/f'validation_end_step{final}_rank{rank}.json');assert end['start_step']==step
 write(w/'resume_acceptance.json',{'status':'passed','optimizer_exact':True,'scheduler_exact':True,'lora_exact':True,'RNG_files_complete':True,'RNG_restoration':'standard Transformers Trainer resume; row schedule verified by actual rank traces','world_size':world,'resume_step':step,'final_step':final})
