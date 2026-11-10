from pathlib import Path
import json,hashlib
CLASSES=[{'id':i,'label':f'synthetic_class_{i}','text':f'SYNTHETIC class {i}'} for i in range(22)]

def create(directory):
    d=Path(directory).resolve();d.mkdir(parents=True,exist_ok=False);rows=[]
    for i in range(10):
        ids=[0] if i%2==0 else [1];empty=i==9;paths=[f'synthetic_{i}.fundus.jpg']+[f'synthetic_{i}_{j:03d}.jpg' for j in range(7,13)]
        rows.append({'case_id':f'case{i}','tie_key':f'case{i:03d}','patient_id':f'synthetic_patient_{i}','center':'synthetic','split':'internal_train' if i<6 else 'val',
            'images':{'cfp':paths[0],'oct':[{'slice':j,'path':paths[j-6]} for j in range(7,13)]},'image_md5':[hashlib.md5(p.encode()).hexdigest() for p in paths],
            'labels':{g:{'ids':[] if empty and g=='oct' else ids,'state':'empty' if empty and g=='oct' else 'disease'} for g in ('cfp','oct','combined')}})
    (d/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8');(d/'texts.json').write_text(json.dumps([{'id':x['id'],'name':x['text'],'definition':{m:f'SYNTHETIC interface example {m} class {x["id"]}; not a clinical description.' for m in 'COB'}} for x in CLASSES],indent=2), encoding='utf-8')
    (d/'classes.json').write_text(json.dumps(CLASSES),encoding='utf-8')
    c={'classes':'classes.json','protocol_id':'synthetic_cpu_only_v1','manifest':'manifest.jsonl','image_root':'.','cache':'cache','output':'results','texts':'texts.json','query_split':'val','gallery_split':'internal_train','protocols':['modal','combined'],'case_directions':['C-C','O-O','B-B','O-C','C-O'],'aggregation':'joint','cohort':'input_available','instruction':'I1','text_kind':'definition','evaluate_text':True,'encoder':{'backend':'synthetic','model':None,'adapter':None,'image_max_tokens':1024,'max_length':8192,'batch_size':2}}
    (d/'config.json').write_text(json.dumps(c,indent=2), encoding='utf-8');return d/'config.json'
