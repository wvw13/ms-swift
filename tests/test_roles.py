from test_delivery import DeliveryTests
from funembed_eval.selection import role_rows,select_rows,encoding_groups
from funembed_eval.encode import execute as encode
from funembed_eval.evaluate import execute as evaluate,audit
from funembed_eval import cache
from pathlib import Path
import numpy as np
class RoleTests(DeliveryTests):
 def test_same_split_smoke_cross_and_text(self):
  _,c,r=self.fixture();c.update(case_protocol='within_split',protocol_id='within_split_smoke',gallery_split='val',smoke_query_limit=2,smoke_gallery_limit=4,case_directions=['C-C','O-C','C-O'],text_modalities=['C'])
  self.assertEqual(len(select_rows(c,r)),4);roles=role_rows(c,r);self.assertEqual([len(roles[k]) for k in ('query','gallery')],[2,4])
  encode(c,r);out=evaluate(c,r);self.assertEqual(audit(c['output'])['tasks_checked'],8)
  self.assertEqual(np.load(Path(c['cache'])/'val_C.npz')['vectors'].shape[0],4)
  self.assertEqual(np.load(Path(c['cache'])/'val_O.npz')['vectors'].shape[0],4)
  for protocol in out['protocols'].values():
   self.assertEqual(protocol['query_counts']['C']['available'],2)
   self.assertEqual(protocol['tasks']['O_C']['exclusions_total']['self'],2)
  c['smoke_query_limit']=3
  with self.assertRaises(ValueError):cache.validate(c['cache'],cache.fingerprint(c,r))
 def test_disjoint_roles_and_all_excluded(self):
  _,c,r=self.fixture();rs=[x for x in r if x['split']=='val'];c.update(case_protocol='within_split',protocol_id='within_split',gallery_split='val',case_directions=['C-C'],evaluate_text=False,query_ids=[rs[0]['case_id']],gallery_ids=[rs[0]['case_id']])
  encode(c,r);out=evaluate(c,r)
  for p in out['protocols'].values():
   self.assertEqual(p['tasks']['C_C']['n_queries_valid'],0);self.assertEqual(p['tasks']['C_C']['n_queries_no_relevant'],1)
 def test_role_ids_preserve_independent_order(self):
  _,c,r=self.fixture();rs=[x for x in r if x['split']=='val'];c.update(gallery_split='val',case_protocol='within_split',query_ids=[rs[3]['case_id'],rs[0]['case_id']],gallery_ids=[rs[1]['case_id'],rs[0]['case_id']])
  roles=role_rows(c,select_rows(c,r));self.assertEqual([x['case_id'] for x in roles['query']],c['query_ids']);self.assertEqual([x['case_id'] for x in roles['gallery']],c['gallery_ids'])
 def test_same_split_exclusions_before_full_relevance_and_idcg(self):
  import math
  from funembed_eval.data import Case
  from funembed_eval.retrieval_core import run_case_retrieval
  q=[Case('q','center','val','val','patient-q',[0,1],'',[],['hash-q'])]
  # Same case, same patient, and shared content must all disappear before ranking.
  g=[Case('q','center','val','val','patient-q',[0,1],'',[],['hash-q']),
     Case('patient-copy','center','val','val','patient-q',[0,1],'',[],['hash-other']),
     Case('image-copy','center','val','val','patient-other',[0,1],'',[],['hash-q']),
     Case('a','center','val','val','patient-a',[0],'',[],['hash-a']),
     Case('b','center','val','val','patient-b',[0,1],'',[],['hash-b'])]
  s=run_case_retrieval(np.array([[1.,0.]]),np.tile([1.,0.],(5,1)),q,g,(1,3,5),'same',str(self.root),{'paths':{'cache_dir':str(self.root)}})
  self.assertEqual(s['exclusions_total'],{'self':1,'same_patient':1,'duplicate_image':1})
  self.assertEqual(s['recall@1'],.5);self.assertEqual(s['recall@5'],1)
  self.assertAlmostEqual(s['ndcg@1'],.5)
  self.assertAlmostEqual(s['ndcg@5'],(.5+1/math.log2(3))/(1+.5/math.log2(3)))
