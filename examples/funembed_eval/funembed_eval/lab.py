"""Generate a local runtime configuration from the privately delivered fixed inputs."""
from pathlib import Path
import json
from .preflight import verify_bundle

def create(a):
    bundle=Path(a.bundle).resolve();verify_bundle(bundle)
    c=json.loads((bundle/'config.template.json').read_text(encoding='utf-8'))
    for key,name in [('manifest','manifest.jsonl'),('texts','texts.json'),('classes','classes.json')]:c[key]=str(bundle/name)
    c['bundle']=str(bundle);run=Path(a.run_dir).resolve()
    c.update(cache=str(run/'encoding'),output=str(run/'results'),query_split=a.query_split,
             case_directions=a.case_directions,text_modalities=a.text_modalities,evaluate_text=bool(a.text_modalities))
    c['encoder'].update(model=str(Path(a.model).resolve()),adapter=str(Path(a.adapter).resolve()) if a.adapter else None,
                        framework_path=str(Path(a.framework_path).resolve()),batch_size=a.batch_size,
                        model_type=a.model_type,representation=a.representation,device=a.device)
    if a.image_root:c['image_root']=str(Path(a.image_root).resolve())
    if a.smoke:
        c.update(protocol_id=c['protocol_id']+'_smoke',smoke_query_limit=2,smoke_gallery_limit=4)
    from .selection import plan
    plan(c)
    p=Path(a.config).resolve();p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(c,f,ensure_ascii=False,indent=2)
    return str(p)
