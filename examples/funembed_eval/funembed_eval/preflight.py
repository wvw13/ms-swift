"""CPU-only metadata and permissions checks; never imports torch or loads weights."""
from pathlib import Path
import importlib.metadata
import json,os
from .data import all_paths,configure_classes,digest,image_paths,label_ids
from .selection import plan,select_rows

def model_contract(c):
    e=c['encoder'];report={'backend':e['backend'],'model_loaded':False}
    if e['backend']=='synthetic':return report
    model=Path(e.get('model') or '')
    if not e.get('model') or not (model/'config.json').is_file():raise ValueError('provide a local full base/merged model directory containing config.json')
    config=json.loads((model/'config.json').read_text(encoding='utf-8'))
    report['architecture']=config.get('architectures');report['model_type_in_config']=config.get('model_type')
    if not list(model.glob('*.safetensors')) and not list(model.glob('pytorch_model*.bin')):raise ValueError('model directory has no local weights')
    saved={}
    for directory in (model,Path(e['adapter']) if e.get('adapter') else model):
        f=directory/'args.json'
        if f.is_file():
            d=json.loads(f.read_text(encoding='utf-8'));saved[str(directory)]={k:d.get(k) for k in ('model_type','task_type','template','train_type','target_modules','modules_to_save') if k in d}
    report['saved_training_metadata']=saved
    if e.get('adapter'):
        a=Path(e['adapter']);ac=json.loads((a/'adapter_config.json').read_text(encoding='utf-8'))
        if not (a/'adapter_model.safetensors').is_file():raise ValueError('adapter_model.safetensors is required for exact loaded-tensor verification')
        if ac.get('peft_type')!='LORA':raise ValueError('this loader currently supports LoRA adapters')
        report['adapter']={k:ac.get(k) for k in ('peft_type','target_modules','modules_to_save','r','lora_alpha')}
    if e['backend']=='swift':
        if e.get('model_type')!='qwen3_vl_emb':raise ValueError('native loader supports model_type=qwen3_vl_emb; provide the actual embedding inference interface for another registration')
        if e.get('representation')!='swift_embedding':raise ValueError('set representation=swift_embedding only for direct last-valid-input-token encoding; generation-conditioned models need their actual inference implementation')
        if config.get('model_type')!='qwen3_vl':raise ValueError('qwen3_vl_emb requires a Qwen3-VL architecture')
        if e.get('template_type','qwen3_vl_emb')!='qwen3_vl_emb':raise ValueError('unverified template: provide a matching inference implementation')
        if e.get('processor') and Path(e['processor'])!=model:raise ValueError('native loader reads processor/tokenizer from model directory; separate customized processor requires an integration adapter')
        for name in ('tokenizer_config.json','preprocessor_config.json'):
            if not (model/name).is_file():raise ValueError(f'model processor is incomplete: {name}')
        report['representation']='last valid INPUT token, final hidden state, L2; no generated tokens'
        report['expected_dimension']=config.get('text_config',{}).get('hidden_size')
    return report

def verify_bundle(path):
    root=Path(path);meta=json.loads((root/'bundle.json').read_text(encoding='utf-8'))
    for name,h in meta['files_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('bundle member escapes directory')
        if digest(p)!=h:raise ValueError(f'bundle checksum mismatch: {name}')
    for name,h in meta.get('code_source_sha256',{}).items():
        p=(Path(__file__).parent/name).resolve()
        if p.parent!=Path(__file__).parent.resolve() or digest(p)!=h:raise ValueError('installed code differs from bundle; use its pinned commit')
    return {'bundle_version':meta['bundle_version'],'code_commit':meta.get('code_commit'),'files_checked':len(meta['files_sha256'])}

def check(c,rows,files=False,environment=False):
    from .encode import read_texts
    classes=configure_classes(c['classes']);bank=read_texts(c)
    if [r['name'] for r in bank]!=[r['text'] for r in classes]:raise ValueError('class names and text IDs differ')
    selected=select_rows(c,rows);p=plan(c)
    report={'protocol_id':c['protocol_id'],'plan':p,'model_loaded':False,'model':model_contract(c),'counts':{}}
    if c.get('bundle'):
        report['bundle']=verify_bundle(c['bundle'])
        bm=json.loads((Path(c['bundle'])/'bundle.json').read_text(encoding='utf-8'))
        for key,filename in [('classes','classes.json'),('texts','texts.json'),('manifest','manifest.jsonl')]:
            if digest(c[key])!=bm['files_sha256'][filename]:raise ValueError(f'{key} differs from the fixed bundle')
    for split,mods in ((c['gallery_split'],p['gallery']),(c['query_split'],p['query'])):
        report['counts'][split]={m:{'available':sum(r['split']==split and bool(image_paths(r,m)) for r in selected),'label_valid':{v:sum(r['split']==split and bool(image_paths(r,m)) and bool(label_ids(r,m,v)) for r in selected) for v in c['protocols']}} for m in mods}
        if any(not x['available'] for x in report['counts'][split].values()):raise ValueError(f'no available images for a requested {split} modality')
    if files:
        seen=set();stamped=0
        for r in selected:
            for i,f in enumerate(all_paths(r)):
                path=Path(f)
                if not path.is_file() or not os.access(path,os.R_OK):raise ValueError('unreadable image; check image_root and directory traversal permissions')
                st=path.stat();seen.add(f)
                if r.get('image_stats'):
                    if r['image_stats'][i]!=[st.st_size,st.st_mtime_ns]:raise ValueError('image changed since checksum bundle; refresh image hashes before retrieval')
                    stamped+=1
        report['image_checks']={'unique_readable_files':len(seen),'checked_stamps':stamped,'image_content_read':False}
    if environment:
        dependencies={}
        names=['numpy'] if c['encoder']['backend']=='synthetic' else ['numpy','torch','torchvision','transformers','peft','accelerate','qwen-vl-utils','Pillow','safetensors']
        for name in names:
            try:dependencies[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:raise ValueError(f'missing dependency {name}; see README installation')
        fw=c['encoder'].get('framework_path')
        if c['encoder']['backend']=='swift':
            if not fw or not (Path(fw)/'swift/infer_engine/transformers_engine.py').is_file():raise ValueError('set framework_path to the compatible ms-swift checkout')
        report['dependencies']=dependencies;report['framework_path']=fw
    return report
