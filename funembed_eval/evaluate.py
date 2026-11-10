"""Selected-task orchestration around the unchanged reference ranking/metric core."""
import csv,json
import numpy as np
from pathlib import Path
from . import cache,retrieval_core as core
from .data import image_paths,label_ids,as_case,configure_classes
from .selection import plan,select_rows,role_rows

def execute(c,rows):
    configure_classes(c['classes']);roles=role_rows(c,rows);rows=select_rows(c,rows);tasks=plan(c)
    meta=cache.validate(c['cache'],cache.fingerprint(c,rows));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    result={'protocol_id':c['protocol_id'],'case_protocol':c['case_protocol'],'query_split':c['query_split'],'gallery_split':c['gallery_split'],'config':c,'cache_fingerprint':meta['fingerprint'],'protocols':{}};long=[];coverage=[]
    for protocol in c['protocols']:
        directory=out/protocol;directory.mkdir()
        available={role:{m:[r for r in roles[role] if image_paths(r,m)] for m in tasks[role]} for role in ('query','gallery')}
        q={m:[r for r in available['query'][m] if label_ids(r,m,protocol)] for m in tasks['query']}
        g={m:[r for r in available['gallery'][m] if label_ids(r,m,protocol)] for m in tasks['gallery']}
        qc={m:[as_case(r,m,protocol) for r in q[m]] for m in q};gc={m:[as_case(r,m,protocol) for r in g[m]] for m in g}
        qv={m:cache.load(c['cache'],f'{c["query_split"]}_{m}.npz',[r['case_id'] for r in q[m]]) for m in q}
        gv={m:cache.load(c['cache'],f'{c["gallery_split"]}_{m}.npz',[r['case_id'] for r in g[m]]) for m in g}
        r={'query_counts':{m:{'available':len(available['query'][m]),'known':len(q[m]),'unknown_or_invalid':len(available['query'][m])-len(q[m])} for m in q},'gallery_counts':{m:len(g[m]) for m in g},'tasks':{}}
        for direction in c['case_directions']:
            src,dst=direction.split('-');tag=src+'_'+dst
            if qv[src].shape[1]!=gv[dst].shape[1]:raise ValueError('query/gallery dimensions differ')
            r['tasks'][tag]=core.run_case_retrieval(qv[src],gv[dst],qc[src],gc[dst],(1,3,5),tag,str(directory),{'paths':{'cache_dir':str(out/'hash_cache')}})
        for m in tasks['text']:
            tag=m+'_text_'+c['text_kind'];tv=cache.load(c['cache'],f'text_{m}.npz',[str(i) for i in range(22)])
            if qv[m].shape[1]!=tv.shape[1]:raise ValueError('image/text dimensions differ')
            r['tasks'][tag]=core.run_text_retrieval(qv[m],tv,qc[m],(1,3,5),tag,str(directory));r['tasks'][tag]['gallery']='22 disease '+c['text_kind']+' prototypes'
        for tag,summary in r['tasks'].items():
            records=[json.loads(line)['metrics'] for line in (directory/f'per_query__{tag}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            counts=[row['n_relevant'] for row in records]
            summary['relevant_candidates_distribution']={'count':len(counts),'zero':sum(n==0 for n in counts),'quantiles':dict(zip(['min','p25','median','p75','max'],np.quantile(counts,[0,.25,.5,.75,1]).tolist())) if counts else {}}
            m=tag[0];n=len(available['query'][m]);known=len(q[m]);valid=summary['n_queries_valid']
            summary.update(n_queries_available=n,n_queries_label_valid=known,n_queries_missing_or_invalid_label=n-known,
                coverage_valid_of_label_valid=valid/known if known else None,coverage_valid_of_available=valid/n if n else None)
            coverage.append({'label_protocol':protocol,'task':tag,**{k:v for k,v in summary.items() if k.startswith(('n_queries','coverage','n_gallery'))}})
            for metric,value in summary.items():
                if '@' in metric:long.append({'protocol_id':c['protocol_id'],'label_protocol':protocol,'split':c['query_split'],'task':tag,'metric':metric,'value':value,'n_queries':summary.get('n_queries_total'),'n_valid':valid,'n_gallery':summary['n_gallery'],'n_no_relevant':summary.get('n_queries_no_relevant')})
        names=['C_C','O_O','B_B']
        if all(x in r['tasks'] and r['tasks'][x].get('ndcg@5') is not None for x in names):
            r['same_modal_ndcg5']={'tasks':names,'aggregation':'equal direction mean','value':sum(r['tasks'][x]['ndcg@5'] for x in names)/3}
        result['protocols'][protocol]=r
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'resolved_config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,records in [('results_long.csv',long),('coverage.csv',coverage)]:
        if records:
            with (out/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'complete.json').write_text(json.dumps({'status':'completed','protocol_id':c['protocol_id'],'summary_sha256':core.ds.digest(out/'summary.json')}),encoding='utf-8')
    return result

def audit(output):
    """Recompute aggregate values from recorded per-query metrics (no re-encoding)."""
    from .metrics import aggregate_query_results
    d=Path(output);summary=json.loads((d/'summary.json').read_text(encoding='utf-8'));checked=0;queries=0
    for protocol,r in summary['protocols'].items():
        for tag,expected in r['tasks'].items():
            rows=[json.loads(x)['metrics'] for x in (d/protocol/f'per_query__{tag}.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
            actual=aggregate_query_results(rows)
            for key,v in actual.items():
                if isinstance(v,(int,float)) and abs(v-expected[key])>1e-12:raise ValueError(f'aggregate differs: {protocol}/{tag}/{key}')
            checked+=1;queries+=len(rows)
    return {'tasks_checked':checked,'query_task_rows':queries,'aggregate_match':True,'recomputed_full_rankings':False}
