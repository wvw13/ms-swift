"""Only requested tasks determine which branches are encoded."""
from .data import image_paths

def plan(c):
    text=list(c.get('text_modalities','COB')) if c.get('evaluate_text',True) else []
    query=set(text);gallery=set()
    for direction in c['case_directions']:
        a,b=direction.split('-');query.add(a);gallery.add(b)
    if not query:raise ValueError('select at least one case direction or text modality')
    return {'query':[m for m in 'COB' if m in query], 'gallery':[m for m in 'COB' if m in gallery], 'text':text}

def role_rows(c,rows):
    p=plan(c);result={}
    for role in ('gallery','query'):
        split=c[role+'_split'];mods=p[role]
        rs=[r for r in rows if r['split']==split and any(image_paths(r,m) for m in mods)]
        if c.get('cohort')=='complete_B':rs=[r for r in rs if image_paths(r,'B')]
        ids=c.get(role+'_ids')
        if ids is not None:
            if len(ids)!=len(set(ids)):raise ValueError('duplicate role IDs')
            available={r['case_id']:r for r in rs}
            if set(ids)-set(available):raise ValueError('role IDs unavailable in split')
            rs=[available[i] for i in ids]
        limit=c.get('smoke_'+role+'_limit')
        if limit:rs=rs[:limit]
        result[role]=rs
    return result

def select_rows(c,rows):
    selected={}
    for rs in role_rows(c,rows).values():
        for r in rs:selected.setdefault(r['case_id'],r)
    return [r for r in rows if r['case_id'] in selected]

def encoding_groups(c,rows):
    roles=role_rows(c,rows);p=plan(c);groups={}
    for role in ('gallery','query'):
        for m in p[role]:
            key=(c[role+'_split'],m);group=groups.setdefault(key,{})
            for r in roles[role]:
                if image_paths(r,m):group.setdefault(r['case_id'],r)
    return [(split,m,list(rs.values())) for (split,m),rs in groups.items()]
