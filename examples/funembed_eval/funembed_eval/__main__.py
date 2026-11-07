import argparse,json,sys
from pathlib import Path
from .config import load
from .data import read_manifest

def main():
    p=argparse.ArgumentParser(description='FunEmbed: fixed-protocol embedding retrieval evaluation')
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('check','encode','evaluate'):
        s=sub.add_parser(command);s.add_argument('--config',required=True)
        for key in ('model','adapter','processor','framework-path','image-root','cache','output','device'):
            s.add_argument('--'+key)
        s.add_argument('--batch-size',type=int);s.add_argument('--case-directions',nargs='*');s.add_argument('--text-modalities',nargs='*')
        if command=='check':s.add_argument('--files',action='store_true');s.add_argument('--environment',action='store_true')
    s=sub.add_parser('demo');s.add_argument('--output',required=True)
    s=sub.add_parser('audit-results');s.add_argument('--output',required=True)
    s=sub.add_parser('verify-bundle');s.add_argument('--bundle',required=True)
    s=sub.add_parser('init-lab',help='Create a configuration; does not load a model or encode images')
    for key in ('bundle','model','framework-path','run-dir','config'):s.add_argument('--'+key,required=True)
    s.add_argument('--adapter');s.add_argument('--image-root');s.add_argument('--batch-size',type=int,default=1)
    s.add_argument('--model-type',required=True);s.add_argument('--representation',required=True)
    s.add_argument('--device',default='cuda:0');s.add_argument('--query-split',choices=('val','internal_test','external_test'),default='val')
    s.add_argument('--case-directions',nargs='*',default=[]);s.add_argument('--text-modalities',nargs='*',default=[]);s.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:
        if a.command=='demo':
            from .demo import create
            print(create(a.output));return
        if a.command=='init-lab':
            from .lab import create
            print(create(a));return
        if a.command=='audit-results':
            from .evaluate import audit
            print(json.dumps(audit(a.output),ensure_ascii=False,indent=2));return
        if a.command=='verify-bundle':
            from .preflight import verify_bundle
            print(json.dumps(verify_bundle(a.bundle),indent=2));return
        excluded={'command','config','files','environment'}
        overrides={k:v for k,v in vars(a).items() if k not in excluded and v is not None}
        if 'text_modalities' in overrides:overrides['evaluate_text']=bool(overrides['text_modalities'])
        c=load(a.config,overrides);rows=read_manifest(c['manifest'],c['image_root'])
        if a.command=='check':
            from .preflight import check
            print(json.dumps(check(c,rows,a.files,a.environment),ensure_ascii=False,indent=2));return
        if a.command=='encode':
            from .preflight import check
            check(c,rows,files=c['encoder']['backend']!='synthetic',environment=True)
            from .encode import execute
        else:
            from .evaluate import execute
        execute(c,rows)
    except (ValueError,FileNotFoundError,PermissionError,FileExistsError,KeyError) as error:
        p.exit(2,f'{a.command}: {error}\n')
if __name__=='__main__':main()
