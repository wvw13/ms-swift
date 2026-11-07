from pathlib import Path
import json

def load(path):
    path=Path(path).resolve();c=json.loads(path.read_text());root=path.parent
    for key in ('manifest','image_root','cache','output','texts'):
        if key in c and c[key] is not None:c[key]=str((root/c[key]).resolve())
    for key in ('model','adapter','processor','framework_path','plugin'):
        if c.get('encoder',{}).get(key):c['encoder'][key]=str((root/c['encoder'][key]).resolve())
    if c['instruction'] not in ('I0','I1'):raise ValueError('instruction must be I0/I1; custom prompts use encoder plugin')
    if c['aggregation'] not in ('joint','per_image'):raise ValueError('unknown aggregation')
    if set(c['protocols'])-{'modal','combined'}:raise ValueError('unknown label protocol')
    if c['gallery_split']!='internal_train':raise ValueError('fixed gallery must be internal_train')
    if c['query_split'] not in ('val','internal_test','external_test'):raise ValueError('invalid query split')
    for d in c['case_directions']:
        if d not in ('C-C','O-O','B-B','O-C','C-O','B-C','B-O','C-B','O-B'):raise ValueError('unknown direction')
    if c['text_kind'] not in ('name','definition'):raise ValueError('name/definition required')
    if c.get('cohort','input_available') not in ('input_available','complete_B'):raise ValueError('unknown cohort')
    e=c['encoder']
    if e['backend'] not in ('swift','legacy','plugin','synthetic'):raise ValueError('unknown encoder backend')
    for key in ('image_max_tokens','max_length','batch_size'):
        if type(e[key])!=int or e[key]<1:raise ValueError(f'invalid {key}')
    if e['backend']=='legacy' and (c['instruction']!='I0' or e.get('adapter')):raise ValueError('legacy backend supports I0 base models only')
    if c['aggregation']=='per_image' and c['instruction']=='I1':raise ValueError('I1 six-slice prompts require joint encoding')
    return c
