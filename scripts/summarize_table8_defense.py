#!/usr/bin/env python3
import argparse, csv, os, math
from pathlib import Path

METHOD_ORDER = ["difgsm","ifgsm","mifgsm","si_ni_fgsm","tifgsm","vit_aware","freq_only","ours"]
METHOD_LABELS = {
 "difgsm":"DI-FGSM","ifgsm":"I-FGSM","mifgsm":"MI-FGSM","si_ni_fgsm":"SI-NI-FGSM",
 "tifgsm":"TI-FGSM","vit_aware":"ViT-Aware","freq_only":"Freq-Only","ours":"Ours (SV-FCA)"
}
TARGET_LABELS = {
 "robustbench:Salman2020Do_R50:imagenet:Linf":"Salman2020Do_R50",
 "robustbench:Mo2022When_ViT-B:imagenet:Linf":"Mo2022When_ViT-B",
 "robustbench:Liu2023Comprehensive_Swin-B:imagenet:Linf":"Liu2023Comprehensive_Swin-B",
}

def read_eval(path):
    if not path.exists(): return {}
    with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    out={}
    for r in rows:
        t=r.get('target','')
        if not t or t=='AVG': continue
        v=r.get('asr') or r.get('asr_clean_correct','')
        try:
            value = float(v)
            if math.isfinite(value): out[t] = value
        except (TypeError, ValueError): pass
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--columns',default='')
    ap.add_argument('--targets', default=','.join(TARGET_LABELS))
    ap.add_argument('--methods', default=','.join(METHOD_ORDER))
    a=ap.parse_args(); root=Path(a.root)
    targets=[t.strip() for t in a.targets.split(',') if t.strip()]
    methods=[m.strip() for m in a.methods.split(',') if m.strip()]
    columns=[c.strip() for c in a.columns.split(',') if c.strip()] if a.columns else [TARGET_LABELS.get(t,t) for t in targets]
    if not targets or not methods or len(columns) != len(targets):
        ap.error('--targets and --methods must be nonempty; --columns must match the number of targets')
    if len(set(columns)) != len(columns) or any(c in ('Attack Method', 'Average') for c in columns):
        ap.error('--columns must be unique and cannot use reserved output column names')
    rows=[]
    for method in methods:
        vals=read_eval(root/method/'eval_results.csv')
        row={'Attack Method':METHOD_LABELS.get(method,method)}; present=[]
        for t, column in zip(targets, columns):
            v=vals.get(t); row[column]='' if v is None else '{:.2f}'.format(v)
            if v is not None: present.append(v)
        row['Average']='' if len(present) != len(targets) else '{:.2f}'.format(sum(present)/len(present)); rows.append(row)
    fields=['Attack Method']+columns+['Average']
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('Saved',a.out)
if __name__=='__main__': main()
