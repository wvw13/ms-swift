"""Only requested tasks determine which branches are encoded."""
from .data import image_paths

def plan(c):
    text=list(c.get('text_modalities','COB')) if c.get('evaluate_text',True) else []
    query=set(text);gallery=set()
    for direction in c['case_directions']:
        a,b=direction.split('-');query.add(a);gallery.add(b)
    if not query:raise ValueError('select at least one case direction or text modality')
    return {'query':[m for m in 'COB' if m in query], 'gallery':[m for m in 'COB' if m in gallery], 'text':text}

def select_rows(c,rows):
    p=plan(c);selected=[]
    for split,mods in ((c['gallery_split'],p['gallery']),(c['query_split'],p['query'])):
        rs=[r for r in rows if r['split']==split and any(image_paths(r,m) for m in mods)]
        if c.get('cohort')=='complete_B':rs=[r for r in rs if image_paths(r,'B')]
        limit=c.get('smoke_query_limit' if split==c['query_split'] else 'smoke_gallery_limit')
        if limit:rs=rs[:limit]
        selected.extend(rs)
    return selected
