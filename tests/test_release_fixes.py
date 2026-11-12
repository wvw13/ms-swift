import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from funembed_train import workflow as wf
from funembed_train.config import write,read,work
from funembed_train.candidates import candidate_steps,check_checkpoint
from funembed_train.lp_summary import summarize
ROOT=Path(__file__).resolve().parents[1]
class ReleaseFixes(unittest.TestCase):
 def cfg(self,tmp):return {'experiment_id':'test','paths':{'output_root':str(tmp),'swift_root':str(tmp)},'training':{'steps':1000,'save_steps':120,'world_size':1,'seed':7},'termination_grace_seconds':.3}
 def test_budget_plans(self):
  c=self.cfg('/tmp')
  for end,expected in [(1000,list(range(120,961,120))+[1000]),(50,[50]),(1200,list(range(120,1201,120)))]:
   c['training']['steps']=end;self.assertEqual(candidate_steps(c),expected)
 def test_explicit_history_subset(self):
  c=read(ROOT/'configs/T1_B_historical.json');self.assertEqual(candidate_steps(c),c['candidate_steps'])
  c=self.cfg('/tmp');c['candidate_steps']=[960,120];self.assertEqual(candidate_steps(c),[120,960])
 def test_invalid_candidates(self):
  for value in [[],[120,120],[0],[1001],[True],[120.0],['120'],[50],None]:
   c=self.cfg('/tmp');c['candidate_steps']=value
   with self.subTest(value=value),self.assertRaises(ValueError):candidate_steps(c)
 def test_locked_callback_final_save(self):
  from transformers.trainer_callback import DefaultFlowCallback,TrainerControl,TrainerState
  from types import SimpleNamespace
  args=SimpleNamespace(logging_first_step=False,logging_strategy='no',eval_strategy='no',save_strategy='steps')
  for end in [50,1000,1200]:
   state=TrainerState(global_step=end,max_steps=end,save_steps=120)
   self.assertTrue(DefaultFlowCallback().on_step_end(args,state,TrainerControl()).should_save)
 def test_checkpoint_complete(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=self.cfg(tmp);p=work(c)/'train/checkpoint-1000';p.mkdir(parents=True)
   with self.assertRaises(ValueError):check_checkpoint(c,1000)
   for f in ['adapter_config.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth']:(p/f).write_bytes(b'x')
   write(p/'trainer_state.json',{'global_step':1000});self.assertEqual(check_checkpoint(c,1000),p)
   write(p/'trainer_state.json',{'global_step':960})
   with self.assertRaises(ValueError):check_checkpoint(c,1000)
 def test_preflight_before_inputs(self):
  c=self.cfg('/tmp');c['candidate_steps']=[25]
  with patch.object(wf,'inputs') as inputs,self.assertRaises(ValueError):wf.prepare(c)
  inputs.assert_not_called()
 def test_seeds_and_export(self):
  from funembed_train.export import export_results,validate_archive
  for encoder in [0,7,None]:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)/'lp/checkpoint-1000/C'
    for seed in [0,1,2]:
     write(root/f'seed{seed}/train_complete.json',{'stopped_early':True})
     write(root/f'seed{seed}/metrics/val.json',{'mAP':.2+seed*.1,'macro_auc':.6,'micro_f1':.3,'macro_f1':.2})
    d=summarize(root,encoder_training_seed=encoder)
    self.assertEqual(d['encoder_training_seed'],encoder);self.assertEqual([h['head_seed'] for h in d['heads']],[0,1,2]);self.assertAlmostEqual(d['aggregate']['val']['mAP']['sample_std'],.1)
    write(Path(tmp)/'resolved.json',{'paths':{},'training':{'seed':encoder}})
    out=Path(tmp)/'result.tar.gz';export_results(Path(tmp),out);validate_archive(out)
    import tarfile
    with tarfile.open(out) as archive:
     name=next(n for n in archive.getnames() if n.endswith('lp/checkpoint-1000/C/summary.json'))
     self.assertEqual(json.load(archive.extractfile(name))['encoder_training_seed'],encoder)
 def test_single_final_selection_export(self):
  from test_select_within import fixture
  from funembed_train.export import export_results,validate_archive
  with tempfile.TemporaryDirectory() as tmp:
   c=self.cfg(tmp);c['training']['steps']=50;c['selection']={'mode':'within_split'}
   cp=work(c)/'train/checkpoint-50';cp.mkdir(parents=True)
   for name in ['adapter_config.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth']:(cp/name).write_bytes(b'synthetic')
   write(cp/'trainer_state.json',{'global_step':50});write(work(c)/'resolved.json',c)
   summary=fixture(work(c)/'evaluation',50)
   selected=wf.select(c,candidate_steps(c),[summary]);self.assertEqual(selected,50)
   out=work(c)/'export.tar.gz';export_results(work(c),out);validate_archive(out)
 def test_workflow_seed_forwarding(self):
  for seed,baseline,step,expected in [(0,False,120,0),(7,False,120,7),(7,True,0,None)]:
   c=self.cfg('/tmp');c['training']['seed']=seed;c['baseline']=baseline
   with patch.object(wf,'run'),patch('funembed_train.lp_summary.summarize') as s:
    wf.lp(c,step,'C','0');self.assertEqual(s.call_args.kwargs['encoder_training_seed'],expected)
 def test_lock_released_on_exception(self):
  import fcntl
  with tempfile.TemporaryDirectory() as tmp:
   c=self.cfg(tmp)
   with patch.object(wf,'_full',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):wf.full(c,'0')
   with (work(c)/'.workflow.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   self.assertEqual(read(work(c)/'status.json')['status'],'interrupted')
 def waitfile(self,p,proc):
  end=time.monotonic()+10
  while time.monotonic()<end:
   if p.exists():return
   if proc.poll() is not None:raise AssertionError('process exited early')
   time.sleep(.03)
  self.fail('readiness timeout')
 def interrupted_tree(self,sig):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);c=self.cfg(tmp);write(d/'cfg.json',c)
   child=d/'child.py';child.write_text('''import subprocess,signal,sys,time,os,json
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])
def stop(s,f):
 p.wait(timeout=5)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid]))
while True:time.sleep(.1)
''')
   parent=d/'parent.py';parent.write_text("from funembed_train.workflow import run\nfrom funembed_train.config import read\nimport sys\nrun(read(sys.argv[1]),'probe',[sys.executable,sys.argv[2],sys.argv[3]])\n")
   env=os.environ|{'PYTHONPATH':str(ROOT)}
   unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'],start_new_session=True)
   with (d/'parent.log').open('w') as log:
    parentp=subprocess.Popen([sys.executable,str(parent),str(d/'cfg.json'),str(child),str(d/'pids.json')],env=env,stdout=log,stderr=log,start_new_session=True)
    try:
     self.waitfile(d/'pids.json',parentp);self.waitfile(work(c)/'stages/probe.json',parentp)
     ids=read(d/'pids.json');os.kill(parentp.pid,sig);parentp.wait(timeout=10)
     for pid in ids:
      with self.assertRaises(ProcessLookupError):os.kill(pid,0)
     self.assertIsNone(unrelated.poll());self.assertEqual(read(work(c)/'stages/probe.json')['status'],'interrupted');self.assertEqual(read(work(c)/'status.json')['status'],'interrupted')
     # Public retry-stage archives only this attempt; then a real retry completes.
     c.update(tasks={'T3':1},directions={'T3':['C-O']});c['training'].update(micro_batch=1,grad_accum=1,effective_query_batch=1);write(d/'cfg.json',c)
     subprocess.run([sys.executable,'-m','funembed_train','retry-stage','--config',str(d/'cfg.json'),'--stage','probe'],env=env,check=True,stdout=log,stderr=log)
     wf.run(c,'probe',[sys.executable,'-c','pass']);self.assertEqual(read(work(c)/'stages/probe.json')['status'],'completed')
    finally:
     if parentp.poll() is None:os.kill(parentp.pid,signal.SIGTERM);parentp.wait(timeout=10)
     unrelated.terminate();unrelated.wait()
 def test_SIGINT_tree_and_retry(self):self.interrupted_tree(signal.SIGINT)
 def test_SIGTERM_tree_and_retry(self):self.interrupted_tree(signal.SIGTERM)
 def test_normal_timeout_handlers(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=self.cfg(tmp);before={s:signal.getsignal(s) for s in [signal.SIGINT,signal.SIGTERM]}
   wf.run(c,'normal',[sys.executable,'-c','pass']);c['stage_timeout_hours']=.0001
   with self.assertRaises(subprocess.TimeoutExpired):wf.run(c,'timeout',[sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(90)'])
   state=read(work(c)/'stages/timeout.json');self.assertEqual(state['status'],'failed');self.assertEqual(state['exit_code'],-9)
   with self.assertRaises(ProcessLookupError):os.kill(state['pid'],0)
   self.assertEqual(before,{s:signal.getsignal(s) for s in before})
if __name__=='__main__':unittest.main()
