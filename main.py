"""Run `python main.py` for the interactive menu or `--help` for commands."""
import argparse
import json
import sys
import sqlite3
import subprocess
from pathlib import Path


def menu():
    from newscheck.workflow import run_project,predict_text,export_sql
    while True:
        print('\nNEWS CHECK | AAI-2602\n1 - Train on Kaggle CSV files\n2 - Run synthetic software demo\n3 - Classify English text\n4 - Export SQL analysis\n5 - Regenerate PDF report and slides\n0 - Exit')
        choice=input('Choose: ').strip()
        try:
            if choice=='0':
                break
            elif choice=='1':
                raw=input('Folder with True.csv and Fake.csv [data/raw]: ').strip() or 'data/raw'
                use_minilm=input('Include pretrained MiniLM (requires extra install/download)? [y/N]: ').lower()=='y'
                run_project(raw,include_minilm=use_minilm)
            elif choice=='2':
                run_project(demo=True)
            elif choice=='3':
                run=input('Completed run folder: ').strip().strip('"')
                text=input('English article text: ')
                print(json.dumps(predict_text(run,text),indent=2))
            elif choice=='4':
                tables=export_sql(input('Run folder: ').strip().strip('"'))
                for name,frame in tables.items():
                    print('\n'+name+'\n'+frame.head(10).to_string(index=False))
            elif choice=='5':
                from newscheck.reporting import build_documents
                build_documents(Path(input('Run folder: ').strip().strip('"')))
            else:
                print('Choose a number from 0 to 5.')
        except (ValueError,OSError,ImportError,RuntimeError,subprocess.SubprocessError,sqlite3.Error) as exc:
            print(f'Operation failed: {exc}')


def main():
    parser=argparse.ArgumentParser(description='Reproducible Fake News Detection project')
    sub=parser.add_subparsers(dest='command')
    run=sub.add_parser('run',help='Train and export all results')
    run.add_argument('--raw',default='data/raw')
    run.add_argument('--out',default='runs')
    run.add_argument('--demo',action='store_true',help='Synthetic software fixture, not research data')
    run.add_argument('--include-minilm',action='store_true')
    run.add_argument('--include-e5',action='store_true')
    pred=sub.add_parser('predict')
    pred.add_argument('--run',required=True)
    pred.add_argument('--text',required=True)
    for command in ['sql','report']:
        cmd=sub.add_parser(command)
        cmd.add_argument('--run',required=True)
    args=parser.parse_args()
    try:
        if args.command=='run':
            from newscheck.workflow import run_project
            run_project(args.raw,args.out,args.demo,args.include_minilm,include_e5=args.include_e5)
        elif args.command=='predict':
            from newscheck.workflow import predict_text
            print(json.dumps(predict_text(args.run,args.text),indent=2))
        elif args.command=='sql':
            from newscheck.workflow import export_sql
            export_sql(args.run)
        elif args.command=='report':
            from newscheck.reporting import build_documents
            build_documents(Path(args.run))
        else:
            menu()
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
