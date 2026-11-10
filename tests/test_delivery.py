import copy,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
from funembed_eval.config import load
from funembed_eval.data import read_manifest,Case
from funembed_eval.demo import create
from funembed_eval.selection import plan,select_rows
from funembed_eval.encode import execute as encode
from funembed_eval.evaluate import execute as evaluate,audit
from funembed_eval.preflight import model_contract
from funembed_eval import metrics,cache,retrieval_core

class DeliveryTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def fixture(self):
        p=create(self.root/'中文目录');c=load(p);return p,c,read_manifest(c['manifest'],c['image_root'])
    def test_only_O_text_no_gallery_or_other_modalities(self):
        p,c,r=self.fixture();c.update(case_directions=[],text_modalities=['O'],evaluate_text=True)
        encode(c,r);out=evaluate(c,r)
        self.assertEqual(sorted(x.name for x in Path(c['cache']).glob('*.npz')),['text_O.npz','val_O.npz'])
        self.assertEqual(out['protocols']['modal']['tasks']['O_text_definition']['n_queries_available'],4)
        self.assertEqual(audit(c['output'])['tasks_checked'],2)
        c['text_modalities']=['C']
        with self.assertRaises(ValueError):cache.validate(c['cache'],cache.fingerprint(c,r))
    def test_only_crossmodal_tasks(self):
        p,c,r=self.fixture();c.update(case_directions=['O-C'],evaluate_text=False)
        encode(c,r);evaluate(c,r)
        self.assertEqual(sorted(x.name for x in Path(c['cache']).glob('*.npz')),['internal_train_C.npz','val_O.npz'])
    def test_all_unknown_labels_and_empty_gallery(self):
        p,c,r=self.fixture();c.update(case_directions=['O-C'],evaluate_text=False)
        for row in r:
            if row['split']=='val':row['labels']['oct']={'ids':[],'state':'empty'}
            else:row['labels']['cfp']={'ids':[],'state':'empty'}
        # Write transformed synthetic inputs so cache identity represents the test dataset.
        Path(c['manifest']).write_text('\n'.join(json.dumps(row) for row in r),encoding='utf-8')
        encode(c,r);result=evaluate(c,r);s=result['protocols']['modal']['tasks']['O_C']
        self.assertEqual(s['n_queries_label_valid'],0);self.assertIsNone(s['coverage_valid_of_label_valid']);self.assertEqual(s['n_gallery'],0)
        self.assertTrue(audit(c['output'])['aggregate_match'])
    def test_no_relevant_visible_in_coverage(self):
        p,c,r=self.fixture();c.update(case_directions=['C-C'],evaluate_text=False)
        for row in r:
            row['labels']['cfp']={'ids':[0 if row['split']=='val' else 1],'state':'disease'}
        encode(c,r);result=evaluate(c,r);s=result['protocols']['modal']['tasks']['C_C']
        self.assertEqual(s['n_queries_no_relevant'],4);self.assertEqual(s['coverage_valid_of_label_valid'],0)
    def test_override_relative_paths_and_subset(self):
        p,c,r=self.fixture();new=load(p,{'output':'relative_results','case_directions':['B-B'],'text_modalities':[],'evaluate_text':False})
        self.assertEqual(new['output'],str(Path('relative_results').resolve()));self.assertEqual(plan(new),{'query':['B'],'gallery':['B'],'text':[]})
        new.update(smoke_query_limit=2,smoke_gallery_limit=3,protocol_id='synthetic_smoke')
        self.assertEqual(len(select_rows(new,r)),5)
    def test_multilabel_and_short_gallery_hand_answers(self):
        # q={0,1}; two valid candidates {0,1} (gain1) and {0} (gain.5).
        q=[Case('q','x','val','val','p0',[0,1],'',[],['q'])]
        g=[Case('b','x','train','internal_train','p1',[0,1],'',[],['b']),Case('a','x','train','internal_train','p2',[0],'',[],['a'])]
        # Equal scores -> a before b; K=5 uses both, no padding with false positives.
        s=retrieval_core.run_case_retrieval(np.array([[1.,0]]),np.array([[1.,0],[1.,0]]),q,g,(1,3,5),'x',str(self.root),{'paths':{'cache_dir':str(self.root)}})
        self.assertEqual(s['hit@1'],1);self.assertEqual(s['recall@1'],.5);self.assertEqual(s['recall@5'],1)
        self.assertAlmostEqual(s['ndcg@5'],(.5+1/math.log2(3))/(1+.5/math.log2(3)))
    def test_stale_image_checksum_rejected(self):
        p,c,r=self.fixture()
        for row in r:
            from funembed_eval.data import all_paths
            row['image_stats']=[]
            for f in all_paths(row):
                Path(f).write_bytes(b'file');st=Path(f).stat();row['image_stats'].append([st.st_size,st.st_mtime_ns])
        Path(r[0]['images']['cfp']).write_bytes(b'changed')
        with self.assertRaises(ValueError):cache.fingerprint(c,r)
    def test_unknown_embedding_semantics_fails(self):
        p,c,r=self.fixture();m=self.root/'model';m.mkdir()
        (m/'config.json').write_text(json.dumps({'model_type':'qwen3_vl','architectures':['Qwen3VLForConditionalGeneration']}),encoding='utf-8')
        (m/'model.safetensors').write_bytes(b'CPU metadata fixture, not real weights')
        c['encoder'].update(backend='swift',model=str(m),model_type='qwen3_vl_emb')
        with self.assertRaisesRegex(ValueError,'representation'):model_contract(c)
    def test_text_dimension_mismatch_fails(self):
        from unittest.mock import patch
        p,c,r=self.fixture();c.update(case_directions=[],text_modalities=['C'])
        class Fake:
            report={'backend':'fixture'}
            def encode(self,branches):
                dim=2 if 'images' in branches[0] else 3
                v=np.ones((len(branches),dim),np.float32)/np.sqrt(dim)
                return v,[-1]*len(v)
        with patch('funembed_eval.encode.backend',return_value=Fake()):
            with self.assertRaisesRegex(ValueError,'dimensions'):encode(c,r)
if __name__=='__main__':unittest.main()
