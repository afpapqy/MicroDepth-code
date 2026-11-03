"""Deterministic SGBM evaluation on an explicit MicroDepth frame list.

No model training or parameter search is performed. Full-mask metrics count
unmatched SGBM predictions as 0px; conditional EPE and prediction coverage are
also reported. The original published data are opened read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from microdepth_tools import Calibration,axial_factor,stereo_error_sums


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    source=args.dataset.resolve();dest=args.output.resolve()
    if dest==source or source in dest.parents:raise ValueError('Output must be outside the original dataset')
    if dest.exists() and any(dest.iterdir()):raise ValueError('Output must be new or empty to preserve prior results')
    dest.mkdir(parents=True,exist_ok=True);pred_dir=dest/'predictions';pred_dir.mkdir(exist_ok=True)
    cv2.setNumThreads(args.threads)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    parameters=dict(manifest['sgbm_predeclared_parameters'])
    create={k:parameters[k] for k in ['minDisparity','numDisparities','blockSize','P1','P2','disp12MaxDiff','preFilterCap','uniquenessRatio','speckleWindowSize','speckleRange']}
    create['mode']=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    matcher=cv2.StereoSGBM_create(**create)
    cal=Calibration.from_json(source/'calibration'/'calibration_20260114_114459.json')
    factor=axial_factor(cal,(960,1280));fB=-cal.P2[0,3]
    paths=[p.strip() for p in args.frames.read_text(encoding='utf-8').splitlines() if p.strip()]
    results=[];acc={};started=time.monotonic();created=datetime.now(timezone.utc).isoformat()
    for i,relative in enumerate(paths,1):
        folder=(source/relative).resolve()
        if source not in folder.parents:raise ValueError('Frame list escapes the dataset root')
        left=cv2.cvtColor(np.asarray(Image.open(folder/'rectified_left.png').convert('RGB')),cv2.COLOR_RGB2GRAY)
        right=cv2.cvtColor(np.asarray(Image.open(folder/'rectified_right.png').convert('RGB')),cv2.COLOR_RGB2GRAY)
        fixed=matcher.compute(left,right)
        pred=np.maximum(fixed.astype(np.float64)/16.,0.)
        returned=fixed>(create['minDisparity']-1)*16
        z=np.load(folder/'depth_gt.npy',allow_pickle=False).astype(np.float64)
        mask=np.asarray(Image.open(folder/'valid_mask.png'))>0
        gt=np.zeros_like(z);gt[mask]=fB*factor[mask]/z[mask]+cal.P1[0,2]-cal.P2[0,2]
        s=stereo_error_sums(pred,gt,mask);matched=mask&returned
        conditional=stereo_error_sums(pred,gt,matched)
        relparts=Path(relative).parts
        category=relparts[1] if relparts[0]=='data' else relparts[0]
        for key in [category,'Overall']:
            a=acc.setdefault(key,{'n':0,'sum_abs':0.,'bad1':0,'bad3':0,'bad5':0,'returned':0,'sum_abs_returned':0.})
            for field,value in s.items():a[field]+=value
            a['returned']+=int(matched.sum());a['sum_abs_returned']+=conditional['sum_abs']
        ident=hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]
        file=pred_dir/f'{ident}.npz';np.savez_compressed(file,disparity_fixed16=fixed)
        results.append({'frame':relative,'category':category,'prediction_file':str(file.relative_to(dest)).replace('\\','/'),
             'prediction_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'valid_pixels':s['n'],'returned_valid_pixels':int(matched.sum()),
             'EPE_full_valid_zero_for_unmatched_px':s['sum_abs']/s['n'],'EPE_returned_only_px':conditional['sum_abs']/conditional['n'] if conditional['n'] else None})
        if i%25==0:
            (dest/'progress.json').write_text(json.dumps({'status':'running','completed_frames':i,'total_frames':len(paths),'elapsed_seconds':round(time.monotonic()-started,1)}),encoding='utf-8')
    metrics=[]
    for key,a in acc.items():
        metrics.append({'category':key,'n_valid':a['n'],'EPE_full_valid_zero_for_unmatched_px':a['sum_abs']/a['n'],
                        'Bad1_pct':100*a['bad1']/a['n'],'Bad3_pct':100*a['bad3']/a['n'],'Bad5_pct':100*a['bad5']/a['n'],
                        'returned_prediction_fraction_pct':100*a['returned']/a['n'],
                        'EPE_returned_only_px':a['sum_abs_returned']/a['returned'] if a['returned'] else None})
    report={'status':'complete','started_utc':created,'completed_utc':datetime.now(timezone.utc).isoformat(),'frames':len(paths),
        'elapsed_seconds':round(time.monotonic()-started,2),'opencv_version':cv2.__version__,'numpy_version':np.__version__,'threads':args.threads,
        'frozen_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'frame_list_sha256':hashlib.sha256(args.frames.read_bytes()).hexdigest(),
        'parameters':parameters,'prediction_encoding':'int16 OpenCV fixed-point disparity; divide by16; (minDisparity-1)*16 denotes unmatched.',
        'label_convention':'Corrected rectified axialZ and rectified left-minus-right disparity; v1.0 valid_mask unchanged.',
        'interpretation':'Classic matching baseline on held-out sessions. No supervised training, no evidence of learnability or cross-specimen generalization.',
        'metrics':metrics,'per_frame':results}
    (dest/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'progress.json').write_text(json.dumps({'status':'complete','completed_frames':len(paths)}),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'per_frame','parameters'}},ensure_ascii=False))


if __name__=='__main__':main()
