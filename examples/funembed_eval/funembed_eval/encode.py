"""Lazy model backends; no CUDA or framework import during CPU checks."""
import importlib.util,json,os,sys
from pathlib import Path
import numpy as np
from .data import image_paths,digest
from .prompts import branch,DEFAULT_INSTRUCTION
from . import cache
from .selection import plan,select_rows,encoding_groups

class SwiftEncoder:
    def __init__(self,c):
        from .preflight import model_contract
        contract=model_contract(c)
        e=c['encoder'];framework=e.get('framework_path')
        if framework:sys.path.insert(0,framework)
        os.environ.update(IMAGE_MAX_TOKEN_NUM=str(e['image_max_tokens']),HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1')
        import torch,swift
        from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
        if framework and Path(swift.__file__).resolve()!=Path(framework,'swift/__init__.py').resolve():raise RuntimeError('unexpected imported swift source')
        if e.get('processor') and Path(e['processor'])!=Path(e['model']):raise ValueError('swift requires colocated processor; separate processor needs plugin')
        self.engine=TransformersEngine(e['model'],adapters=[e['adapter']] if e.get('adapter') else None,
            model_type=e['model_type'],template_type=e.get('template_type','qwen3_vl_emb'),task_type='embedding',torch_dtype=torch.bfloat16,attn_impl='sdpa',device_map=e.get('device','cuda:0'),max_batch_size=e['batch_size'])
        self.engine.template.max_length=e['max_length'];self.engine.template.truncation_strategy='raise'
        self.engine.model.requires_grad_(False);self.engine.model.eval()
        if e.get('adapter'):
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            expected=load_file(str(Path(e['adapter'])/'adapter_model.safetensors'));actual=get_peft_model_state_dict(self.engine.model)
            if set(actual)!=set(expected) or not all(torch.equal(actual[k].detach().cpu(),expected[k].to(actual[k].dtype)) for k in expected):raise RuntimeError('adapter tensor verification failed')
            count=len(expected)
        else:
            if any('lora_' in n for n,_ in self.engine.model.named_parameters()):raise RuntimeError('unexpected LoRA')
            count=0
        self.report={'backend':'swift','contract':contract,'adapter_tensors_checked':count,'swift_source':str(Path(swift.__file__).resolve()),'extraction':'last valid input token, final hidden state, L2; no generation'}
        self.Request=InferRequest;self.RequestConfig=RequestConfig
    def encode(self,branches):
        out=self.engine.infer([self.Request(**b) for b in branches],self.RequestConfig(),use_tqdm=False)
        if len(out)!=len(branches):raise ValueError('response count mismatch')
        for r in out:
            if isinstance(r,Exception):raise r
        return np.asarray([r.data[0].embedding for r in out],np.float32),[r.usage.prompt_tokens for r in out]
class SyntheticEncoder:
    report={'backend':'synthetic','extraction':'deterministic hash vectors, no clinical/model meaning'}
    def encode(self,branches):
        v=np.asarray([np.random.default_rng(int(cache.stable(b)[:16],16)).normal(size=32) for b in branches],np.float32)
        return v/np.linalg.norm(v,axis=1,keepdims=True),[-1]*len(branches)
def backend(c):
    e=c['encoder']
    if e['backend']=='swift':return SwiftEncoder(c)
    if e['backend']=='synthetic':return SyntheticEncoder()
    if e['backend']=='plugin':
        if not all(e.get(k) for k in ('plugin','representation_protocol')):raise ValueError('plugin and representation_protocol required')
        spec=importlib.util.spec_from_file_location('funembed_user_encoder',e['plugin']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.create_encoder(c)
        if not callable(getattr(model,'encode',None)) or not isinstance(getattr(model,'report',None),dict):raise ValueError('plugin needs encode(branches) and report')
        return model
    from .legacy_encoders import load_encoder
    settings=dict(e);settings.update(path=e['model'],type=e['model_type'],device='cuda:0',dtype=e.get('dtype','bfloat16'),instruction=DEFAULT_INSTRUCTION,max_pixels=e['image_max_tokens']*32*32,min_pixels=4096,joint_batch_size=e['batch_size'],embed_dim=e.get('embed_dim',2048))
    return load_encoder(e['model_type'],{'models':{e['model_type']:settings}})
def read_texts(c):
    bank=json.loads(Path(c['texts']).read_text(encoding='utf-8'))
    if [r['id'] for r in bank]!=list(range(22)):raise ValueError('full ordered 22-class text bank required')
    for r in bank:
        for m in 'COB':
            v=r['name'] if c['text_kind']=='name' else r['definition'][m]
            if not isinstance(v,str) or not v.strip():raise ValueError('missing candidate text')
    return bank

def execute(c,rows):
    rows=select_rows(c,rows);tasks=plan(c)
    directory=Path(c['cache']);fp=cache.fingerprint(c,rows)
    if (directory/'complete.json').exists():cache.validate(directory,fp);print('REUSE',directory);return
    directory.mkdir(parents=True,exist_ok=False);(directory/'fingerprint.json').write_text(json.dumps(fp,indent=2), encoding='utf-8')
    enc=backend(c);e=c['encoder'];batch=e['batch_size'];bank=read_texts(c)
    dimension=None
    def encode_items(items,name,ids,mod,text=False):
        nonlocal dimension
        vectors=[];tokens=[]
        for start in range(0,len(items),batch):
            part=items[start:start+batch]
            if e['backend']=='legacy':
                if text:v=enc.encode_texts(part)
                elif c['aggregation']=='joint' and mod!='C':v=enc.encode_joint(part)
                elif mod=='C':v=enc.encode_images([x[0] for x in part])
                else:
                    av=enc.encode_images([p for images in part for p in images]).reshape(len(part),len(part[0]),-1)
                    if mod=='O':v=av.mean(1)
                    else:o=av[:,1:].mean(1);o/=np.linalg.norm(o,axis=1,keepdims=True);v=.5*av[:,0]+.5*o
                t=[-1]*len(v)
            elif c['aggregation']=='per_image' and not text and mod!='C':
                av,ti=enc.encode([branch(c,mod,images=[p]) for images in part for p in images]);av/=np.linalg.norm(av,axis=1,keepdims=True);av=av.reshape(len(part),len(part[0]),-1)
                if mod=='O':v=av.mean(1)
                else:o=av[:,1:].mean(1);o/=np.linalg.norm(o,axis=1,keepdims=True);v=.5*av[:,0]+.5*o
                t=np.asarray(ti).reshape(len(part),-1).sum(1).tolist()
            else:v,t=enc.encode([branch(c,mod,text=x) if text else branch(c,mod,images=x) for x in part])
            if c['aggregation']=='per_image' and not text and mod!='C':v=v/np.linalg.norm(v,axis=1,keepdims=True)
            v=np.asarray(v,np.float32)
            if v.ndim!=2 or not np.isfinite(v).all():raise ValueError('encoder returned non-finite or non-matrix embeddings')
            if dimension is None:dimension=v.shape[1]
            if dimension!=v.shape[1]:raise ValueError('image/text embedding dimensions differ')
            if len(v)!=len(part) or len(t)!=len(part):raise ValueError('encoder lost rows')
            if c['aggregation']=='joint' and any(n>e['max_length'] for n in t):raise ValueError('sequence exceeds max_length')
            vectors.extend(v);tokens.extend(t);print(name,min(start+batch,len(items)),len(items),flush=True)
        if not ids:raise ValueError('empty encoding set: '+name)
        cache.save(directory,name,vectors,ids,tokens)
    legacy_done=set()
    for split,mod,rs in encoding_groups(c,rows):
        mods=[m for sp,m,_ in encoding_groups(c,rows) if sp==split]
        if split in legacy_done:continue
        if e['backend']=='legacy' and c['aggregation']=='per_image' and c.get('cohort')=='complete_B' and c['query_split']!=c['gallery_split']:
            from .retrieval_core import aggregate
            rs=[r for r in rows if r['split']==split and image_paths(r,'B')]
            av=enc.encode_images([p for r in rs for p in image_paths(r,'B')]).reshape(len(rs),7,-1)
            for m in mods:cache.save(directory,f'{split}_{m}.npz',aggregate(av,m),[r['case_id'] for r in rs],[-1]*len(rs))
            legacy_done.add(split)
            continue

        encode_items([image_paths(r,mod) for r in rs],f'{split}_{mod}.npz',[r['case_id'] for r in rs],mod)
    if c.get('evaluate_text',True):
        for mod in tasks['text']:encode_items([r['name'] if c['text_kind']=='name' else r['definition'][mod] for r in bank],f'text_{mod}.npz',[str(i) for i in range(22)],mod,True)
    report=enc.info() if e['backend']=='legacy' else enc.report
    if report.get('truncated'):raise ValueError('legacy encoder reported truncation')
    (directory/'resolved_config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'encoder_report.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    (directory/'complete.json').write_text(json.dumps({'fingerprint':cache.stable(fp),'artifacts':{p.name:digest(p) for p in directory.glob('*.npz')}},indent=2), encoding='utf-8')
