import csv,json
from pathlib import Path
from . import cache,retrieval_core as core
from .data import image_paths,label_ids,as_case

def execute(c,rows):
    meta=cache.validate(c['cache'],cache.fingerprint(c,rows));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    result={'protocol_id':c['protocol_id'],'config':c,'cache_fingerprint':meta['fingerprint'],'protocols':{}};long=[]
    for protocol in c['protocols']:
        directory=out/protocol;directory.mkdir()
        available={s:{m:[r for r in rows if r['split']==s and image_paths(r,m)] for m in 'COB'} for s in (c['query_split'],c['gallery_split'])}
        q={m:[r for r in available[c['query_split']][m] if label_ids(r,m,protocol)] for m in 'COB'};g={m:[r for r in available[c['gallery_split']][m] if label_ids(r,m,protocol)] for m in 'COB'}
        qc={m:[as_case(r,m,protocol) for r in q[m]] for m in 'COB'};gc={m:[as_case(r,m,protocol) for r in g[m]] for m in 'COB'}
        qv={m:cache.load(c['cache'],f'{c["query_split"]}_{m}.npz',[r['case_id'] for r in q[m]]) for m in 'COB'};gv={m:cache.load(c['cache'],f'{c["gallery_split"]}_{m}.npz',[r['case_id'] for r in g[m]]) for m in 'COB'}
        r={'query_counts':{m:{'available':len(available[c['query_split']][m]),'known':len(q[m]),'unknown_or_invalid':len(available[c['query_split']][m])-len(q[m])} for m in 'COB'},'gallery_counts':{m:len(g[m]) for m in 'COB'},'tasks':{}}
        for direction in c['case_directions']:
            src,dst=direction.split('-');tag=src+'_'+dst
            r['tasks'][tag]=core.run_case_retrieval(qv[src],gv[dst],qc[src],gc[dst],(1,3,5),tag,str(directory),{'paths':{'cache_dir':str(out/'hash_cache')}})
        if c.get('evaluate_text',True):
            for m in 'COB':
                tag=m+'_text_'+c['text_kind'];tv=cache.load(c['cache'],f'text_{m}.npz',[str(i) for i in range(22)])
                r['tasks'][tag]=core.run_text_retrieval(qv[m],tv,qc[m],(1,3,5),tag,str(directory));r['tasks'][tag]['gallery']='22 disease '+c['text_kind']+' prototypes'
        result['protocols'][protocol]=r
        for tag,summary in r['tasks'].items():
            for metric,value in summary.items():
                if '@' in metric:long.append({'protocol_id':c['protocol_id'],'label_protocol':protocol,'split':c['query_split'],'task':tag,'metric':metric,'value':value,'n_queries':summary.get('n_queries_total'),'n_valid':summary.get('n_queries_valid'),'n_gallery':summary['n_gallery'],'n_no_relevant':summary.get('n_queries_no_relevant')})
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if long:
        with (out/'results_long.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(long[0]));w.writeheader();w.writerows(long)
    (out/'complete.json').write_text(json.dumps({'status':'completed','protocol_id':c['protocol_id'],'summary_sha256':core.ds.digest(out/'summary.json')}))
    return result
