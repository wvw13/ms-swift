import unittest,tempfile,json,copy
from pathlib import Path
from funembed_train.select_within import select,stable,sha

def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d),encoding='utf-8')
def fixture(root,step,scores=(.3,.6,.9),batch=1,split='val',ids=None):
 d=root/f'checkpoint-{step}';ca=d/'cache';ca.mkdir(parents=True);rs=d/'results';rs.mkdir()
 ids=ids or ['a','b'];e={'adapter':str(root/f'checkpoint-{step}'),'model':'synthetic-base','batch_size':batch,'image_max_tokens':1024,'max_length':8192,'device':'cuda:0'}
 c={'cache':str(ca),'encoder':e,'protocol_id':'synthetic_shorttest','case_protocol':'within_split','query_split':split,'gallery_split':split,'instruction':'I1','aggregation':'joint','cohort':'input_available','case_directions':['C-C','O-O','B-B'],'evaluate_text':False,'text_kind':'definition','protocols':['modal']}
 manifest=ca/'manifest.jsonl';manifest.write_text('\n'.join(json.dumps({'case_id':i,'tie_key':i,'patient_id':i,'center':'synthetic','split':split,'images':{'cfp':i+'.jpg','oct':[{'slice':n,'path':i+str(n)+'.jpg'} for n in range(7,13)]},'labels':{g:{'ids':[0],'state':'disease'} for g in ('cfp','oct','combined')}}) for i in ids),encoding='utf-8')
 c.update(manifest=str(manifest),image_root=str(root),classes=str(ca/'classes.json'),texts=str(ca/'texts.json'))
 write(ca/'classes.json',[]);write(ca/'texts.json',[])
 fp={k:c[k] for k in ('encoder','instruction','aggregation','cohort','case_directions','evaluate_text','text_kind','case_protocol')};fp.update(splits=[split,split],roles={'query':ids,'gallery':ids},encoding_groups={f'val_{m}':ids for m in 'COB'},manifest_sha256='synthetic_manifest',adapter_sha256={'adapter':'hash'+str(step)})
 fp.update({k+'_sha256':sha(c[k]) for k in ('manifest','classes','texts')})
 artifacts={}
 for m in 'COB':
  f=ca/f'val_{m}.npz';f.write_bytes(b'synthetic bytes: selector tests hashes, not encodes');artifacts[f.name]=sha(f)
 write(ca/'fingerprint.json',fp);write(ca/'complete.json',{'fingerprint':stable(fp),'artifacts':artifacts})
 tasks={}
 for m,score in zip('COB',scores):
  tasks[m+'_'+m]={'ndcg@5':score,'n_queries_valid':2,'n_queries_label_valid':2,'n_queries_no_relevant':0,'n_queries_available':2}
  f=rs/'modal'/f'per_query__{m}_{m}.jsonl';f.parent.mkdir(exist_ok=True);f.write_text('\n'.join(json.dumps({'metrics':{'ndcg@5':score,'no_relevant':False,'sample_path':i}}) for i in ids),encoding='utf-8')
 s={'config':c,'case_protocol':'within_split','protocol_id':'synthetic_shorttest','query_split':split,'gallery_split':split,'cache_fingerprint':stable(fp),'protocols':{'modal':{'tasks':tasks,'query_counts':{m:{'available':2,'known':2} for m in 'COB'},'gallery_counts':{m:2 for m in 'COB'}}}}
 f=rs/'summary.json';write(f,s);write(rs/'complete.json',{'status':'completed','summary_sha256':sha(f)});return f
class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_mean_tie(self):
  a=fixture(self.root,8);b=fixture(self.root,16);r=select([b,a]);self.assertAlmostEqual(r['selected']['score'],.6);self.assertEqual(r['selected']['step'],8)
 def test_better(self):
  a=fixture(self.root,8);b=fixture(self.root,16,(.6,.6,.9));self.assertEqual(select([a,b])['selected']['step'],16)
 def test_batch_mismatch(self):
  a=fixture(self.root,8);b=fixture(self.root,16,batch=4)
  with self.assertRaises(ValueError):select([a,b])
 def test_roles_mismatch(self):
  a=fixture(self.root,8);b=fixture(self.root,16,ids=['a','c'])
  with self.assertRaises(ValueError):select([a,b])
 def test_test_split_refused(self):
  a=fixture(self.root,8);b=fixture(self.root,16,split='internal_test')
  with self.assertRaises(ValueError):select([a,b])
 def test_missing_complete(self):
  a=fixture(self.root,8);b=fixture(self.root,16);(b.parent/'complete.json').unlink()
  with self.assertRaises(FileNotFoundError):select([a,b])
 def test_changed_scores(self):
  a=fixture(self.root,8);b=fixture(self.root,16);d=json.loads(b.read_text());d['protocols']['modal']['tasks']['C_C']['ndcg@5']=.9;write(b,d);write(b.parent/'complete.json',{'status':'completed','summary_sha256':sha(b)})
  with self.assertRaises(ValueError):select([a,b])
 def test_missing_direction(self):
  a=fixture(self.root,8);b=fixture(self.root,16);d=json.loads(b.read_text());del d['protocols']['modal']['tasks']['O_O'];write(b,d);write(b.parent/'complete.json',{'status':'completed','summary_sha256':sha(b)})
  with self.assertRaises(ValueError):select([a,b])
 def test_changed_artifact(self):
  a=fixture(self.root,8);b=fixture(self.root,16);(b.parent.parent/'cache/val_C.npz').write_bytes(b'changed')
  with self.assertRaises(ValueError):select([a,b])
 def test_bad_query_ids(self):
  a=fixture(self.root,8);b=fixture(self.root,16);f=b.parent/'modal/per_query__C_C.jsonl';f.write_text(f.read_text().replace('"sample_path": "a"','"sample_path": "wrong"'))
  with self.assertRaises(ValueError):select([a,b])
if __name__=='__main__':unittest.main()
