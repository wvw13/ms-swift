import unittest,tempfile,json,math,copy
from pathlib import Path
import numpy as np
from funembed_eval import metrics as m,retrieval_core as core,cache
from funembed_eval.data import Case,read_manifest,image_paths,label_ids
from funembed_eval.config import load
from funembed_eval.demo import create
from funembed_eval.prompts import branch
from funembed_eval.encode import execute as encode
from funembed_eval.evaluate import execute as evaluate

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def fixture(self):
        path=create(self.root/'demo');c=load(path);return c,read_manifest(c['manifest'],c['image_root'])
    def test_cli_config_and_order(self):
        c,rs=self.fixture();r=rs[0]
        self.assertEqual([Path(p).name for p in image_paths(r,'B')],[f'synthetic_0.fundus.jpg']+[f'synthetic_0_{i:03d}.jpg' for i in range(7,13)])
        b=branch(c,'B',images=image_paths(r,'B'));self.assertEqual(b['messages'][1]['content'],'<image>'*7)
        self.assertNotIn(r['patient_id'],str(b));self.assertNotIn('diabetic_retinopathy',str(b))
        r['images']['oct'].pop();self.assertEqual(len(image_paths(r,'C')),1);self.assertFalse(image_paths(r,'O'));self.assertFalse(image_paths(r,'B'))
    def test_modal_empty_not_normal(self):
        c,rs=self.fixture();self.assertEqual(label_ids(rs[-1],'O','modal'),[]);self.assertEqual(label_ids(rs[-1],'O','combined'),[1])
    def test_bad_slice_and_normal_conflict(self):
        c,rs=self.fixture();p=Path(c['manifest']);lines=[json.loads(s) for s in p.read_text().splitlines()]
        lines[0]['images']['oct'].reverse();p.write_text('\n'.join(json.dumps(r) for r in lines))
        with self.assertRaises(ValueError):read_manifest(p,c['image_root'])
        lines[0]['images']['oct'].reverse();lines[0]['labels']['cfp']={'ids':[0,21],'state':'disease'};p.write_text('\n'.join(json.dumps(r) for r in lines))
        with self.assertRaises(ValueError):read_manifest(p,c['image_root'])
    def test_ties(self):
        order=m.rank_candidates(np.array([1.,1.,.5]),np.array([True,True,True]),np.array([1,0,2]));self.assertEqual(order.tolist(),[1,0,2])
    def test_metrics_linear_full_denominator(self):
        r=m.evaluate_query(np.array([.5,1.,0]),np.array([.5,1.,.25]),(1,3,5))
        self.assertAlmostEqual(r['recall@1'],1/3);self.assertAlmostEqual(r['ndcg@1'],.5)
        self.assertAlmostEqual(r['ndcg@3'],(.5+1/math.log2(3))/(1+.5/math.log2(3)+.25/2))
    def test_exclusions_priority(self):
        def case(k,p,h):return Case(k,'x','x','val',p,[0],'',[],[h])
        q=[case('same','a','h1')];g=[case('same','a','h1'),case('p','a','h2'),case('d','b','h1'),case('ok','c','h3')]
        mask,reasons=core.exclusion_masks(q,g,{'paths':{'cache_dir':str(self.root)}})
        self.assertEqual(mask.tolist(),[[False,False,False,True]]);self.assertEqual(reasons,{'self':1,'same_patient':1,'duplicate_image':1})
    def test_norel_mean_excludes(self):
        a=m.evaluate_query(np.array([1.]),np.array([1.]),(1,3,5));b=m.evaluate_query(np.array([0.]),np.array([0.]),(1,3,5));r=m.aggregate_query_results([a,b])
        self.assertEqual(r['ndcg@5'],1);self.assertEqual(r['n_queries_no_relevant'],1)
    def test_end_to_end_and_cache_invalidation(self):
        c,rs=self.fixture();encode(c,rs);r=evaluate(c,rs)
        self.assertEqual(r['protocols']['modal']['query_counts']['O']['known'],3)
        self.assertEqual(r['protocols']['combined']['query_counts']['O']['known'],4)
        self.assertEqual(len(r['protocols']['modal']['tasks']),8)
        self.assertTrue((Path(c['output'])/'results_long.csv').exists())
        changed=copy.deepcopy(c);changed['instruction']='I0'
        with self.assertRaises(ValueError):cache.validate(c['cache'],cache.fingerprint(changed,rs))
        p=Path(c['cache'])/'val_C.npz';p.write_bytes(p.read_bytes()+b'altered')
        with self.assertRaises(ValueError):cache.validate(c['cache'],cache.fingerprint(c,rs))
    def test_legacy_seven_image_batch_order(self):
        from unittest.mock import patch
        c,rs=self.fixture();c.update(instruction='I0',aggregation='per_image',cohort='complete_B',evaluate_text=False)
        c['encoder'].update(backend='legacy',model_type='clip',dtype='float32')
        from funembed_eval.data import all_paths
        for r in rs:
            for p in all_paths(r):Path(p).write_bytes(b'synthetic file for stat only')
        calls=[]
        class Fake:
            def encode_images(self,paths):
                calls.append(paths)
                return np.tile(np.array([[.6,.8]],np.float32),(len(paths),1))
            def info(self):return {'backend':'fake legacy CPU'}
        with patch('funembed_eval.encode.backend',return_value=Fake()):encode(c,rs)
        self.assertEqual(calls[0],[p for r in rs[:6] for p in image_paths(r,'B')])
        self.assertEqual(len(calls),2)
        values=cache.load(c['cache'],'internal_train_B.npz',[r['case_id'] for r in rs[:6]])
        np.testing.assert_allclose(values,np.tile([[.6,.8]],(6,1)),atol=1e-7)
    def test_aggregation(self):
        v=np.array([[[1.,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]])
        np.testing.assert_allclose(core.aggregate(v,'O'),[[0,1]])
        np.testing.assert_allclose(core.aggregate(v,'B'),[[1/2**.5,1/2**.5]])
if __name__=='__main__':unittest.main()
