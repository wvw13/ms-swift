from pathlib import Path
import hashlib,json,importlib.metadata
import numpy as np
from .data import digest,all_paths
from .selection import select_rows,plan,role_rows,encoding_groups

def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fingerprint(c,rows):
    rows=select_rows(c,rows)
    e=c['encoder'];model=Path(e.get('model') or '/nonexistent');adapter=Path(e['adapter']) if e.get('adapter') else None
    weights={str(p.relative_to(model)):[p.stat().st_size,p.stat().st_mtime_ns] for p in list(model.glob('*.safetensors'))+list(model.glob('*.bin'))}
    configs={p.name:digest(p) for p in model.glob('*.json')}
    adapter_hash={p.name:digest(p) for p in adapter.iterdir() if p.name in ('adapter_config.json','adapter_model.safetensors','adapter_model.bin')} if adapter else {}
    files={}
    for r in rows:
        for i,p in enumerate(all_paths(r)):
            f=Path(p)
            if f.exists():
                s=f.stat();files[p]=[s.st_size,s.st_mtime_ns]
                if r.get('image_stats') and r['image_stats'][i]!=files[p]:raise ValueError('image changed since checksum bundle; refresh checksums')
            elif e['backend']!='synthetic':raise FileNotFoundError(f)
    runtime={}
    for name in ('numpy','torch','transformers','peft','ms-swift','qwen-vl-utils'):
        try:runtime[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:runtime[name]=None
    framework=Path(e['framework_path']) if e.get('framework_path') else None
    framework_sha={str(p.relative_to(framework)):digest(p) for p in framework.glob('swift/**/*.py')} if framework else {}
    plugin_hash=digest(e['plugin']) if e.get('plugin') else None
    processor=Path(e['processor']) if e.get('processor') else model
    return {'schema':3,'case_protocol':c.get('case_protocol','reference_train'),'roles':{role:[r['case_id'] for r in rs] for role,rs in role_rows(c,rows).items()},'encoding_groups':{split+'_'+m:[r['case_id'] for r in rs] for split,m,rs in encoding_groups(c,rows)},'bundle_sha256':digest(Path(c['bundle'])/'bundle.json') if c.get('bundle') else None,'tasks':plan(c),'case_directions':c['case_directions'],'selected_case_ids':[r['case_id'] for r in rows],'classes_sha256':digest(c['classes']),'splits':[c['gallery_split'],c['query_split']],'evaluate_text':c.get('evaluate_text',True),'encoder':e,'instruction':c['instruction'],'aggregation':c['aggregation'],'cohort':c.get('cohort','input_available'),
        'manifest_sha256':digest(c['manifest']),'texts_sha256':digest(c['texts']),'text_kind':c['text_kind'],
        'image_stat':files,'base_weight_stat':weights,'base_config_sha256':configs,'model_scripts_sha256':{str(p.relative_to(model)):digest(p) for p in model.glob('scripts/*.py')},'chat_template_sha256':{p.name:digest(p) for p in model.glob('*.jinja')},'adapter_sha256':adapter_hash,
        'processor_sha256':{p.name:digest(p) for p in processor.glob('*.json')},'plugin_sha256':plugin_hash,
        'framework_source_sha256':framework_sha,'runtime':runtime,
        'package_source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
def save(directory,name,vectors,ids,tokens):
    directory=Path(directory);v=np.asarray(vectors,dtype=np.float32)
    if v.ndim!=2 or len(v)!=len(ids) or len(set(ids))!=len(ids) or not np.isfinite(v).all():raise ValueError('invalid embedding output')
    if np.any(np.linalg.norm(v,axis=1)==0):raise ValueError('zero embedding')
    if not np.allclose(np.linalg.norm(v,axis=1),1,atol=.01):raise ValueError('encoder must return L2-normalized vectors')
    np.savez_compressed(directory/name,vectors=v,ids=np.asarray(ids),tokens=np.asarray(tokens))
def load(directory,name,ids):
    with np.load(Path(directory)/name,allow_pickle=False) as d:
        lookup={str(key):i for i,key in enumerate(d['ids'])};v=d['vectors']
        if v.ndim!=2:raise ValueError('cached embeddings must be a matrix')
        if len(lookup)!=len(v) or not np.isfinite(v).all() or (np.linalg.norm(v,axis=1)==0).any():raise ValueError('invalid cached embeddings')
        if len(set(ids))!=len(ids):raise ValueError('duplicate requested IDs')
        if any(i not in lookup for i in ids):raise ValueError('cache missing requested case IDs')
        if not np.allclose(np.linalg.norm(v,axis=1),1,atol=.01):raise ValueError('cache vectors are not normalized')
        return v[np.asarray([lookup[i] for i in ids],dtype=int)].copy()
def validate(directory,fp):
    d=Path(directory);meta=json.loads((d/'complete.json').read_text(encoding='utf-8'))
    if meta['fingerprint']!=stable(fp) or json.loads((d/'fingerprint.json').read_text(encoding='utf-8'))!=fp:raise ValueError('cache fingerprint mismatch; use a new cache directory')
    for name,h in meta['artifacts'].items():
        if digest(d/name)!=h:raise ValueError('cache artifact changed: '+name)
    return meta
