import io,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from funembed_train.config import write,read,work
from funembed_train import workflow as wf
from funembed_train import export as ex
class ExportRecoveryTests(unittest.TestCase):
 def fixture(self,folder):
  c={'experiment_id':'export_test','paths':{'output_root':str(folder)},'baseline':True,'training':{},'lp_modalities':[]};r=work(c);write(r/'resolved.json',c);return c,r
 def test_pack_failure_preserves_partial_not_final(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);out=r/'result_export.tar.gz'
   with patch.object(tarfile.TarFile,'add',side_effect=OSError('disk full')):
    with self.assertRaises(OSError):ex.export_results(r,out)
   self.assertFalse(out.exists());self.assertTrue(list(r.glob('*.partial')));ex.export_results(r,out);ex.validate_archive(out)
 def test_validation_failure_not_published(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);out=r/'result_export.tar.gz'
   with patch.object(ex,'validate_archive',side_effect=ex.InvalidArchive('injected check failure')):
    with self.assertRaises(ex.InvalidArchive):ex.export_results(r,out)
   self.assertFalse(out.exists());self.assertTrue(list(r.glob('*.partial')))
 def test_retry_only_export(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp)
   with patch.object(wf,'inputs'),patch.object(wf,'evaluation') as evaluate:
    with patch.object(ex,'validate_archive',side_effect=ex.InvalidArchive('failed check')):
     with self.assertRaises(ex.InvalidArchive):wf.full(c,'0')
    self.assertEqual(evaluate.call_count,3);self.assertEqual(read(r/'status.json')['stage'],'export');self.assertEqual(read(r/'status.json')['status'],'failed')
   with patch.object(wf,'inputs',side_effect=AssertionError('repeated inputs')),patch.object(wf,'evaluation',side_effect=AssertionError('repeated encoding')),patch.object(wf,'train',side_effect=AssertionError('repeated training')),patch.object(wf,'lp',side_effect=AssertionError('repeated LP')):wf.full(c,'0')
   self.assertEqual(read(r/'status.json')['status'],'completed');ex.validate_archive(r/'result_export.tar.gz')
 def test_completed_corruption_preserved_then_repaired(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);write(r/'status.json',{'status':'completed'});out=r/'result_export.tar.gz';out.write_bytes(b'broken legacy export')
   with patch.object(wf,'inputs',side_effect=AssertionError('rerun')):result=wf.full(c,'0')
   self.assertEqual(Path(result['preserved_invalid']).read_bytes(),b'broken legacy export');ex.validate_archive(out)
 def test_completed_reuse_validates_without_overwrite(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);write(r/'export_ready.json',{'compute_stages_complete':True});wf.full(c,'0');out=r/'result_export.tar.gz';before=out.stat().st_mtime_ns
   with patch.object(wf,'inputs',side_effect=AssertionError('rerun')),patch.object(ex,'validate_archive',wraps=ex.validate_archive) as validate:result=wf.full(c,'0');self.assertGreater(validate.call_count,0)
   self.assertTrue(result['reused']);self.assertEqual(out.stat().st_mtime_ns,before)
 def test_valid_different_export_not_overwritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);out=r/'result_export.tar.gz';first=ex.export_results(r,out);write(r/'environment.json',{'new':'metadata'})
   with self.assertRaises(FileExistsError):ex.export_results(r,out,recover_invalid=True)
   self.assertEqual(ex.validate_archive(out)['sha256'],first['sha256'])
 def test_truncated_gzip_and_unsafe_path_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);out=r/'result_export.tar.gz';ex.export_results(r,out);out.write_bytes(out.read_bytes()[:-8])
   with self.assertRaises(ex.InvalidArchive):ex.validate_archive(out)
   evil=r/'evil.tar.gz'
   with tarfile.open(evil,'w:gz') as tar:
    m=tarfile.TarInfo('funembed_results/../../escaped');m.size=1;tar.addfile(m,io.BytesIO(b'x'))
   with self.assertRaises(ex.InvalidArchive):ex.validate_archive(evil)
   self.assertFalse((r.parent/'escaped').exists())
 def test_config_changed_refuses_recovery(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);write(r/'export_ready.json',{'compute_stages_complete':True});c['training']['steps']=9
   with self.assertRaises(ValueError):wf.full(c,'0')
 def test_hash_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);out=r/'bad.tar.gz';folder=r/'funembed_results';write(folder/'resolved_config.json',{});write(folder/'export_policy.json',{});write(folder/'FILES.sha256.json',{'resolved_config.json':'bad','export_policy.json':'bad'})
   with tarfile.open(out,'w:gz') as tar:tar.add(folder,arcname='funembed_results')
   with self.assertRaises(ex.InvalidArchive):ex.validate_archive(out)
 def test_python_311_rejected_before_work(self):
  import subprocess,sys
  result=subprocess.run([sys.executable,'-c',"import sys;sys.version_info=(3,11,2);import funembed_train"],capture_output=True,text=True)
  self.assertNotEqual(result.returncode,0);self.assertIn('requires Python >=3.13',result.stderr)
 def test_smoke_selected_checkpoint_test_and_lp_entry(self):
  with tempfile.TemporaryDirectory() as tmp:
   c,r=self.fixture(tmp);c.update(baseline=False,smoke=True,training={'steps':4,'save_steps':2},selection={'mode':'within_split'},candidate_steps=[2,4]);write(r/'resolved.json',c)
   with patch.object(wf,'prepare'),patch.object(wf,'train'),patch.object(wf,'env',return_value={}),patch('funembed_train.acceptance.check_training'),patch.object(wf,'evaluation',return_value=Path('synthetic')) as evaluate,patch.object(wf,'select',return_value=2),patch.object(wf,'lp') as lp:
    wf.full(c,'0,1')
   self.assertEqual([(x.args[1],x.args[2]) for x in evaluate.call_args_list],[(2,'val'),(4,'val'),(2,'internal_test'),(2,'external_test')]);self.assertEqual(lp.call_args.args[4],'all');self.assertTrue(lp.call_args.kwargs['tiny'])
