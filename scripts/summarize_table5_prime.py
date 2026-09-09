#!/usr/bin/env python3
"""Build final Table V for SV-FCA from per-run eval_results.csv files."""
import argparse, csv, os, math
from pathlib import Path

VARIANT_LABELS = {
    "full_model": "Full Model (SV-FCA)",
    "without_sv_pool": "Without SV Pool",
    "without_frequency_coordination": "Without Frequency Coordination",
    "without_band_consensus": "Without Band Consensus",
    "without_low_mid_prior": "Without Low-Mid Prior",
    "without_spectral_memory": "Without Spectral Memory",
}
SETTING_LABELS = {
    "cnn_to_vit": "CNN-to-ViT",
    "vit_to_cnn": "ViT-to-CNN",
    "vit_to_vit": "ViT-to-ViT",
    "mixed_mixed": "Mixed-to-Mixed",
}

def read_avg_metric(path, metric):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows: return None
    avg = next((r for r in rows if r.get('target') == 'AVG'), None)
    if avg is None:
        raise ValueError(f"Missing pooled AVG row in {path}")
    value = avg.get(metric, '')
    if value in (None, '') and metric == 'asr': value = avg.get('asr_clean_correct', '')
    if value in (None, ''):
        return None
    result = float(value)
    return result if math.isfinite(result) else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--metric', default='asr', choices=['asr', 'asr_clean_correct', 'asr_all', 'adv_acc_all', 'clean_acc', 'robust_acc_clean_correct'])
    ap.add_argument('--settings', default='cnn_to_vit,vit_to_cnn,vit_to_vit,mixed_mixed')
    ap.add_argument('--variants', default='full_model,without_sv_pool,without_frequency_coordination,without_band_consensus,without_low_mid_prior,without_spectral_memory')
    a=ap.parse_args()
    settings=[x.strip() for x in a.settings.split(',') if x.strip()]
    variants=[x.strip() for x in a.variants.split(',') if x.strip()]
    rows=[]
    for v in variants:
        row={'Variant':VARIANT_LABELS.get(v,v)}; vals=[]
        for setting in settings:
            p=Path(a.root)/setting/v/'eval_results.csv'
            val=read_avg_metric(p,a.metric) if p.exists() else None
            row[SETTING_LABELS.get(setting,setting)]='' if val is None else '{:.2f}'.format(val)
            if val is not None: vals.append(val)
        row['Average']='' if len(vals) != len(settings) or not vals else '{:.2f}'.format(sum(vals)/len(vals))
        rows.append(row)
    fields=['Variant']+[SETTING_LABELS.get(x,x) for x in settings]+['Average']
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('Saved',a.out)
if __name__=='__main__': main()
