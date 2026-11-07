import argparse,json
from .config import load
from .data import read_manifest,image_paths

def main():
    p=argparse.ArgumentParser(description='FunEmbed portable retrieval: encode, cached evaluate, CPU fixtures');sub=p.add_subparsers(dest='command',required=True)
    for command in ('check','encode','evaluate'):
        s=sub.add_parser(command);s.add_argument('--config',required=True)
    s=sub.add_parser('demo');s.add_argument('--output',required=True);a=p.parse_args()
    if a.command=='demo':
        from .demo import create
        print(create(a.output));return
    c=load(a.config);rows=read_manifest(c['manifest'],c['image_root'])
    if c.get('cohort')=='complete_B':rows=[r for r in rows if image_paths(r,'B')]
    if a.command=='check':
        from .encode import read_texts
        read_texts(c);print(json.dumps({'protocol_id':c['protocol_id'],'backend':c['encoder']['backend'],'counts':{s:{m:sum(r['split']==s and bool(image_paths(r,m)) for r in rows) for m in 'COB'} for s in (c['gallery_split'],c['query_split'])},'model_loaded':False},indent=2));return
    if a.command=='encode':
        from .encode import execute
    else:
        from .evaluate import execute
    execute(c,rows)
if __name__=='__main__':main()
