from pathlib import Path
import json,hashlib
from .config import read,write,sha,work
from funembed_eval.data import read_manifest,image_paths,all_paths,configure_classes
SPLITS=['internal_train','val','internal_test','external_test']
def prepare(c,files=True):
 w=work(c);src=Path(c['paths']['private_input_root']);paths={k:src/f for k,f in [('manifest','manifest.jsonl'),('classes','classes.json'),('texts','texts.json')]}
 for p in paths.values():
  if not p.is_file():raise FileNotFoundError('Required private input: '+str(p))
 identities={k:sha(p) for k,p in paths.items()};target=w/'inputs'
 if (target/'complete.json').exists():
  previous=read(target/'complete.json')
  if previous['source_sha256']!=identities or previous['data_root']!=c['paths']['data_root']:raise ValueError('input provenance changed: use a new experiment directory')
  if files and not previous.get('files_checked'):raise ValueError('previous preparation did not verify files: use a new output directory')
  return previous
 classes=configure_classes(paths['classes']);texts=read(paths['texts']);assert [t['id'] for t in texts]==list(range(22))
 for x in texts:
  assert x['name'] and all(isinstance(x['definition'][m],str) and x['definition'][m].strip() for m in 'COB')
  expected=f"CFP features: {x['definition']['C'].strip()} OCT features: {x['definition']['O'].strip()}"
  if x['definition']['B']!=expected:raise ValueError('B text does not match fixed CFP/OCT composition')
 rows=read_manifest(paths['manifest'],c['paths']['data_root']);patients={s:set() for s in SPLITS};counts={s:{m:0 for m in 'COB'} for s in SPLITS};classrows={s:[] for s in SPLITS};train={m:[] for m in 'COB'};hash_checks=0
 for r in rows:
  key=json.dumps([r['center'],r['patient_id']],separators=(',',':'));patients[r['split']].add(key)
  if not r.get('image_md5'):raise ValueError('image MD5 required for exclusion; supply audited manifest')
  for i,p in enumerate(all_paths(r)):
   if not files:continue
   st=Path(p).stat()
   if not r.get('image_stats') or r['image_stats'][i]!=[st.st_size,st.st_mtime_ns]:
    if hashlib.md5(Path(p).read_bytes()).hexdigest()!=r['image_md5'][i]:raise ValueError('migrated image content differs')
    hash_checks+=1
  if files:r['image_stats']=[[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in all_paths(r)]
  for m in 'COB':
   images=image_paths(r,m)
   if not images:continue
   counts[r['split']][m]+=1
   if r['split']=='internal_train':
    if not r.get('input_ids',{}).get(m):raise ValueError('missing original input_id '+m)
    train[m].append(dict(case_id=r['case_id'],input_id=r['input_ids'][m],input_type=m,patient_key=key,patient_anon=hashlib.sha256(key.encode()).hexdigest()[:16],center=r['center'],split=r['split'],use_in_training_allowed=True,image_paths=images,labels=r['labels']))
  if image_paths(r,'B') and r['labels']['combined']['state'] in ('normal','disease'):
   classrows[r['split']].append(dict(record_id=r['case_id'],case_id=r['case_id'],patient_anon=hashlib.sha256(key.encode()).hexdigest()[:16],center=r['center'],split=r['split'],images={m:image_paths(r,m) for m in 'COB'},labels=[int(i in r['labels']['combined']['ids']) for i in range(22)]))
 for i,s in enumerate(SPLITS):
  for t in SPLITS[i+1:]:
   if patients[s]&patients[t]:raise ValueError('patient crosses splits')
 target.mkdir(parents=True,exist_ok=True)
 for k in ['classes','texts']:write(target/(k+'.json'),read(paths[k]))
 with (target/'manifest.jsonl').open('w',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 phase=target/'phase2';(phase/'internal_train').mkdir(parents=True,exist_ok=True)
 for m,rs in train.items():
  (phase/'internal_train'/f'{m}.input_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
 write(phase/'text_dictionary.json',{'entries':[{'label_id':t['id'],'english_text':t['name']} for t in texts]})
 for m in 'CO':write(target/f'definition_{m}.json',{cl['label']:texts[cl['id']]['definition'][m] for cl in classes})
 cls=w/'classification/data/manifests';cls.mkdir(parents=True,exist_ok=True);hashes={}
 for split,rs in classrows.items():
  f=cls/f'{split}.jsonl';f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8');hashes[split]=sha(f)
 sizes={s:len(v) for s,v in classrows.items()};expected=c.get('expected_classification_counts')
 if expected and sizes!=expected:raise ValueError('classification cohort count mismatch: '+str(sizes))
 write(cls/'summary.json',{'counts':sizes,'manifest_sha256':hashes,'source_sha256':identities,'class_map':classes})
 cfg=read(Path(__file__).with_name('classification_config.json'));cfg['models']={'2b':c['paths']['model_root']};write(w/'classification/configs/common.json',cfg)
 result={'source_sha256':identities,'data_root':c['paths']['data_root'],'counts':counts,'classification_counts':sizes,'migrated_image_MD5_checks':hash_checks,'files_checked':files};write(target/'complete.json',result);return result
