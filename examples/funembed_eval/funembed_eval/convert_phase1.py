"""Convert an existing audited phase1 manifest; does not discover or read images."""
import argparse,json
from pathlib import Path
from .data import CLASSES,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--image-root',required=True);p.add_argument('--output',required=True);p.add_argument('--cfp-definitions',required=True);p.add_argument('--oct-definitions',required=True);p.add_argument('--variant',type=int);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);root=Path(a.image_root).resolve();seen=set();count=0
    def relative(path):return str(Path(path).resolve().relative_to(root)) if path else None
    with (out/'manifest.jsonl').open('w') as f:
        for line in Path(a.manifest).open():
            r=json.loads(line)
            if r['case_id'] in seen:raise ValueError('duplicate case_id: resolve duplicate records explicitly before conversion')
            seen.add(r['case_id']);im=r['images']
            item={'case_id':r['case_id'],'tie_key':r['sample_path'],'patient_id':r['patient_key'],'center':r['center'],'split':r['split'],
                'images':{'cfp':relative(im['cfp_path']),'oct':[{'slice':x['slice'],'path':relative(x['path'])} for x in im['oct'] if 7<=x['slice']<=12]},
                'labels':{g:{'ids':r['labels'][g]['ids'],'state':r['labels'][g]['state']} for g in ('cfp','oct','combined')}}
            f.write(json.dumps(item,ensure_ascii=False)+'\n');count+=1
    defs={m:json.loads(Path(path).read_text()) for m,path in [('C',a.cfp_definitions),('O',a.oct_definitions)]}
    for m,d in defs.items():
        if set(d)!={r['label'] for r in CLASSES}:raise ValueError('definition dictionary must have canonical 22 keys')
        for key,v in d.items():
            if isinstance(v,list):
                if a.variant is None:raise ValueError('list definitions require explicit --variant')
                v=v[a.variant]
            if not isinstance(v,str) or not v.strip():raise ValueError('empty definition')
            d[key]=v.strip()
    bank=[]
    for r in CLASSES:
        c,o=defs['C'][r['label']],defs['O'][r['label']]
        bank.append({'id':r['id'],'name':r['text'],'definition':{'C':c,'O':o,'B':f'CFP features: {c} OCT features: {o}'}})
    (out/'texts.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2))
    (out/'conversion.json').write_text(json.dumps({'records':count,'source_sha256':{str(Path(p).resolve()):digest(p) for p in (a.manifest,a.cfp_definitions,a.oct_definitions)},'image_root':str(root),'image_content_read':False},indent=2))
    print(json.dumps({'records':count,'output':str(out)}))
if __name__=='__main__':main()
