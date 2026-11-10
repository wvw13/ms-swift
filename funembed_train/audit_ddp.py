"""Check actual rank traces and summarize bounded DDP throughput/memory."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'vendor/v1'))
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from common import read_json,read_jsonl,dump


def summarize(root,pairs,run):
    cfg=read_json(pairs/'config.json');world=cfg['world_size'];ga=cfg['grad_accum'];micro=cfg['micro_batch']
    sched=read_jsonl(pairs/'schedule.jsonl');seen=[];perstep=defaultdict(list);memory={}
    for rank in range(world):
        trace=read_jsonl(run/f'actual_microsteps_rank{rank}.jsonl')
        end=read_json(run/f'validation_end_step{cfg["steps"]}_rank{rank}.json')
        grads=read_jsonl(run/f'gradient_audit_rank{rank}.jsonl')
        assert end['status']=='passed' and end['changed_lora_tensors']>0
        assert len(trace)==cfg['steps']*ga and len(grads)==cfg['steps']
        assert all(x['finite'] and x['nonzero_gradient_tensors']>0 for x in grads)
        for t in trace:
            expected_step=sched[t['step']-1]
            assert (t['task'],t['direction'])==(expected_step['task'],expected_step['direction'])
            first=((t['step']-1)*ga+t['micro'])*world*micro+rank
            assert t['rows']==[first+j*world for j in range(micro)]
            assert t['accumulation_divisor']==ga and np.isfinite(t['loss']) and t['loss']>0
            assert max(t['branch_lengths'])<8192
            seen+=t['rows'];perstep[t['step']].append(t)
        updates=read_jsonl(run/f'completed_updates_rank{rank}.jsonl')
        assert [x['global_step'] for x in updates]==list(range(1,cfg['steps']+1))
        memory[str(rank)]={'allocated_GiB':max(x['peak_allocated_GiB'] for x in updates),
                           'reserved_GiB':max(x['peak_reserved_GiB'] for x in updates)}
    assert sorted(seen)==list(range(cfg['rows'])) and len(seen)==len(set(seen))
    times=read_jsonl(run/'completed_updates.jsonl');last=0.;by_direction=defaultdict(list)
    for update in times:
        delta=update['step_seconds'];last=update['elapsed_seconds']
        tr=perstep[update['global_step']]
        assert len({(x['task'],x['direction']) for x in tr})==1
        nonpad=sum(x['nonpadding_tokens'] for x in tr);padded=sum(x['padded_tokens'] for x in tr)
        by_direction[update['direction']].append({'step':update['global_step'],'seconds':delta,
            'queries':world*micro*ga,'images':sum(x['image_count'] for x in tr),
            'nonpadding_tokens':nonpad,'padded_tokens':padded,'padding_fraction':1-nonpad/padded,
            'max_sequence_length':max(max(x['branch_lengths']) for x in tr)})
    out={'status':'passed','config':cfg,'memory_per_rank':memory,'rows_verified':len(seen),
         'direction_updates':dict(Counter(s['direction'] for s in sched)),'directions':{}}
    for direction,entries in by_direction.items():
        stable=entries[1:] if len(entries)>1 else entries
        seconds=float(np.median([e['seconds'] for e in stable]))
        out['directions'][direction]={'updates':entries,'stable_update_median_seconds':seconds,
            'stable_query_throughput':world*micro*ga/seconds,'stable_samples':len(stable)}
    dump(root/(run.name+'_summary.json'),out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();summarize(a.root,a.pairs,a.run)
