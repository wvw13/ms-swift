"""Paths resolve relative to the config; CLI overrides resolve relative to cwd."""
from pathlib import Path
import json
from .data import configure_classes
from .selection import plan

PATHS=('manifest','image_root','cache','output','texts','classes','bundle')
ENCODER_PATHS=('model','adapter','processor','framework_path','plugin')

def load(path,overrides=None):
    path=Path(path).resolve();c=json.loads(path.read_text(encoding='utf-8'));root=path.parent
    for key in PATHS:
        if c.get(key):c[key]=str((root/c[key]).resolve())
    for key in ENCODER_PATHS:
        if c.get('encoder',{}).get(key):c['encoder'][key]=str((root/c['encoder'][key]).resolve())
    for key,value in (overrides or {}).items():
        if value is None:continue
        if key in ENCODER_PATHS or key in ('batch_size','device'):
            c['encoder'][key]=str(Path(value).resolve()) if key in ENCODER_PATHS else value
        elif key in PATHS:c[key]=str(Path(value).resolve())
        else:c[key]=value
    for key in ('manifest','image_root','cache','output','texts','classes'):
        if not c.get(key):raise ValueError(f'missing configuration field: {key}')
    configure_classes(c['classes'])
    if c['instruction'] not in ('I0','I1'):raise ValueError('instruction must be I0/I1')
    if c['aggregation'] not in ('joint','per_image'):raise ValueError('unknown aggregation')
    if not c['protocols'] or set(c['protocols'])-{'modal','combined'}:raise ValueError('unknown label protocol')
    c.setdefault('case_protocol','reference_train')
    if c['case_protocol']=='reference_train':
        if c['gallery_split']!='internal_train':raise ValueError('reference_train requires internal_train gallery')
    elif c['case_protocol']=='within_split':
        if c['gallery_split']!=c['query_split'] or 'within_split' not in c['protocol_id']:raise ValueError('within_split needs matching splits and distinct protocol_id')
    else:raise ValueError('unknown case_protocol')
    if c['query_split'] not in ('val','internal_test','external_test'):raise ValueError('invalid query split')
    if len(c['case_directions'])!=len(set(c['case_directions'])):raise ValueError('duplicate direction')
    for d in c['case_directions']:
        if d not in ('C-C','O-O','B-B','O-C','C-O','B-C','B-O','C-B','O-B'):raise ValueError('unknown direction')
    if c['text_kind'] not in ('name','definition'):raise ValueError('name/definition required')
    c.setdefault('text_modalities',list('COB'))
    if set(c['text_modalities'])-set('COB') or len(set(c['text_modalities']))!=len(c['text_modalities']):raise ValueError('invalid text modalities')
    if c.get('cohort','input_available') not in ('input_available','complete_B'):raise ValueError('unknown cohort')
    e=c['encoder']
    if e['backend'] not in ('swift','legacy','plugin','synthetic'):raise ValueError('unknown encoder backend')
    for key in ('image_max_tokens','max_length','batch_size'):
        if type(e[key])!=int or e[key]<1:raise ValueError(f'invalid {key}')
    for key in ('smoke_query_limit','smoke_gallery_limit'):
        if key in c and (type(c[key])!=int or c[key]<1 or 'smoke' not in c['protocol_id']):raise ValueError('limited samples require a smoke protocol_id')
    if e['backend']=='legacy' and (c['instruction']!='I0' or e.get('adapter')):raise ValueError('legacy supports I0 base models only')
    if c['aggregation']=='per_image' and c['instruction']=='I1':raise ValueError('I1 six-slice prompts require joint encoding')
    if Path(c['cache'])==Path(c['output']):raise ValueError('cache and output must be different directories')
    plan(c)
    return c
