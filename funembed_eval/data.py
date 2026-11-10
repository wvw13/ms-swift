"""Portable manifest and explicit image/label policy; no patient-key inference."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,re

N_CLASSES=22
CLASSES=[{'id':i,'label':f'synthetic_class_{i}','text':f'SYNTHETIC class {i}'} for i in range(N_CLASSES)]

def configure_classes(path):
    global CLASSES
    rows=json.loads(Path(path).read_text(encoding='utf-8'))
    if [r['id'] for r in rows]!=list(range(N_CLASSES)):raise ValueError('classes must contain ordered IDs 0..21')
    if len({r['label'] for r in rows})!=N_CLASSES:raise ValueError('duplicate canonical class name')
    if any(not isinstance(r.get('text'),str) or not r['text'].strip() for r in rows):raise ValueError('missing English name')
    CLASSES=rows
    return rows
GROUP={'C':'cfp','O':'oct','B':'combined'}
@dataclass
class Case:
    sample_path:str
    center:str
    part:str
    project_split:str
    patient_key:str
    labels:list
    cfp_path:str
    oct_paths:list
    md5s:list|None=None
    @property
    def image_paths(self):return ([self.cfp_path] if self.cfp_path else [])+self.oct_paths

def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_manifest(path,image_root):
    rows=[];seen=set();root=Path(image_root)
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        r=json.loads(line)
        for k in ('case_id','tie_key','patient_id','center','split','images','labels'):
            if k not in r:raise ValueError(f'missing manifest field: {k}')
        if not all(isinstance(r[k],str) and r[k] for k in ('case_id','tie_key','patient_id','center','split')):raise ValueError('empty identity field')
        if r['case_id'] in seen:raise ValueError('duplicate case_id')
        seen.add(r['case_id'])
        if r['split'] not in ('internal_train','val','internal_test','external_test'):raise ValueError('unknown split')
        if r['center']=='shandong' and r['split']!='external_test':raise ValueError('Shandong must be external_test')
        imgs=r['images'];octs=imgs.get('oct',[]);slices=[x['slice'] for x in octs]
        if slices!=sorted(set(slices)) or any(n not in range(7,13) for n in slices):raise ValueError('central OCT must be unique, ordered 007..012')
        def resolve(p):return str((root/p).resolve()) if p else None
        r['images']={'cfp':resolve(imgs.get('cfp')),'oct':[{'slice':x['slice'],'path':resolve(x['path'])} for x in octs]}
        for group in GROUP.values():
            lab=r['labels'][group];ids=lab['ids'];state=lab['state']
            if ids!=sorted(set(ids)) or any(type(i)!=int or i<0 or i>=22 for i in ids):raise ValueError('invalid label IDs')
            if state not in ('normal','disease','empty','unknown','conflict'):raise ValueError('unknown label state')
            if state=='normal' and ids!=[21]:raise ValueError('normal must be explicit [21]')
            if state=='disease' and (not ids or 21 in ids):raise ValueError('disease/normal conflict')
            if state=='empty' and ids:raise ValueError('empty state with labels')
        hashes=r.get('image_md5')
        stamps=r.get('image_stats')
        if stamps is not None and len(stamps)!=len(all_paths(r)):raise ValueError('image_stats length mismatch')
        if hashes is not None:
            if len(hashes)!=len(all_paths(r)) or any(not re.fullmatch('[0-9a-f]{32}',h) for h in hashes):raise ValueError('hashes must cover all available CFP/central OCT')
        rows.append(r)
    if len({r['tie_key'] for r in rows})!=len(rows):raise ValueError('duplicate tie_key')
    return rows

def all_paths(r):return ([r['images']['cfp']] if r['images']['cfp'] else [])+[x['path'] for x in r['images']['oct']]
def image_paths(r,mod):
    c=r['images']['cfp'];o=r['images']['oct'];complete=[x['slice'] for x in o]==list(range(7,13))
    if mod=='C':return [c] if c else []
    if mod=='O':return [x['path'] for x in o] if complete else []
    if mod=='B':return [c]+[x['path'] for x in o] if c and complete else []
    raise ValueError(mod)
def label_ids(r,mod,protocol):
    group=GROUP[mod] if protocol=='modal' else 'combined'
    lab=r['labels'][group]
    return lab['ids'] if lab['state'] in ('normal','disease') else []
def as_case(r,mod,protocol):
    return Case(r['tie_key'],r['center'],r['split'],r['split'],r['patient_id'],label_ids(r,mod,protocol),r['images']['cfp'],[x['path'] for x in r['images']['oct']],r.get('image_md5'))
def attach_md5s(cases,cfg):
    """Hash every available case image, including modalities not encoded."""
    path=Path(cfg['paths']['cache_dir'])/'image_md5.json';path.parent.mkdir(parents=True,exist_ok=True)
    cache=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for case in cases:
        if case.md5s is not None:continue
        values=[]
        for image in case.image_paths:
            st=Path(image).stat();key=str(Path(image).resolve());entry=cache.get(key,{})
            stamp=[st.st_size,st.st_mtime_ns]
            if entry.get('stat')!=stamp:entry={'stat':stamp,'md5':digest(image,'md5')};cache[key]=entry
            values.append(entry['md5'])
        case.md5s=values
    path.write_text(json.dumps(cache,sort_keys=True), encoding='utf-8')
