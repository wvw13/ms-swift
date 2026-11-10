import unittest,tempfile,json,tarfile,copy,os,subprocess,sys
from pathlib import Path
from collections import Counter
from funembed_train.synthetic import fixture,demo
from funembed_train import workflow as wf
from funembed_train.config import work,read,load,write,TASKS
from funembed_train.inputs import prepare
class TrainingTests(unittest.TestCase):
 def test_all_directions_pairs_schedule_export(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp)/'fixture';r=demo(d);self.assertFalse(r['GPU_used']);self.assertGreater(r['evaluation_audit']['tasks_checked'],0)
   c=read(d/'config.json');w=work(c);s=[json.loads(l) for l in (w/'pairs/schedule.jsonl').read_text().splitlines()];self.assertEqual(Counter((x['task'],x['direction']) for x in s),Counter({(t,x):2 for t,ds in TASKS.items() for x in ds}))
   with tarfile.open(w/'cpu_results.tar.gz') as tar:
    text=''.join(tar.extractfile(m).read().decode() for m in tar.getmembers() if m.isfile());self.assertNotIn('synthetic_internal_train_0',text);self.assertNotIn('synthetic_val_0',text);self.assertFalse(any(m.name.endswith(('.pt','.npz','.safetensors','.jpg')) for m in tar.getmembers()))
 def test_independent_t3_and_single_directions(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=fixture(Path(tmp)/'f')
   for task,ds in TASKS.items():
    for direction in ds:
     x=copy.deepcopy(c);x.update(experiment_id=task+'_'+direction,tasks={task:1},directions={task:[direction]});x['training']['steps']=2;wf.prepare(x);s=[json.loads(l) for l in (work(x)/'pairs/schedule.jsonl').read_text().splitlines()];self.assertEqual({z['direction'] for z in s},{direction})
 def test_input_changes_and_missing_files(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=fixture(Path(tmp)/'f');prepare(c);f=Path(c['paths']['private_input_root'])/'texts.json';x=read(f);x[0]['name']='changed';write(f,x)
   with self.assertRaises(ValueError):prepare(c)
 def test_missing_text_class_and_image_content(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=fixture(Path(tmp)/'f');f=next(Path(c['paths']['data_root']).glob('*'));f.write_bytes(b'changed')
   with self.assertRaises(ValueError):prepare(c)
 def test_custom_tasks_and_bad_directions(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=fixture(Path(tmp)/'f');c['directions']['T3']=['C-C'];p=Path(tmp)/'cfg.json';write(p,c)
   with self.assertRaises(ValueError):load(p)
 def test_lp_head_statistics(self):
  from funembed_train.lp_summary import summarize
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for seed,value in enumerate([.2,.4,.6]):
    d=root/f'seed{seed}';write(d/'train_complete.json',{'stopped_early':True});write(d/'logs/epoch_0001.json',{'epoch':1,'val_mAP':value})
    write(d/'metrics/val.json',{'mAP':value,'macro_auc':.8,'micro_f1':.6,'macro_f1':.5,'threshold':.3})
   d=summarize(root);self.assertAlmostEqual(d['aggregate']['val']['mAP']['mean'],.4);self.assertAlmostEqual(d['aggregate']['val']['mAP']['sample_std'],.2);self.assertEqual(d['aggregate']['val']['mAP']['n_heads'],3)
if __name__=='__main__':unittest.main()
