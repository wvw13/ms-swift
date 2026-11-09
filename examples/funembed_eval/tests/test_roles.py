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
