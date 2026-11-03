"""Evaluate cached predictions without altering data or predictions.

Example:
 python evaluate_cached.py --data DATA/MicroDepth --stereo FFS_ROOT --mono DA3_ROOT \
   --partition test --out NEW_OUTPUT_DIRECTORY

Prediction directories use the retained legacy category/session/frame layout;
frame_paths.csv maps that layout to the public release hierarchy. No inference,
training, ROI restriction or implicit invalid-prediction removal occurs here.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np
from microdepth_tools import Calibration,corrected_rectified_z,depth_to_disparity,mono_error_sums,stereo_error_sums

PACKAGE=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data',required=True,type=Path);parser.add_argument('--stereo',type=Path);parser.add_argument('--mono',type=Path)
 parser.add_argument('--partition',choices=['all','train','validation','test'],default='test')
 parser.add_argument('--convention',choices=['corrected','legacy'],default='corrected')
 parser.add_argument('--mapping',type=Path,default=PACKAGE/'frame_paths.csv');parser.add_argument('--out',required=True,type=Path)
 args=parser.parse_args()
 if not args.stereo and not args.mono:parser.error('Supply at least one prediction root')
 out=args.out.resolve();roots=[p.resolve() for p in [args.data,args.stereo,args.mono] if p]
 if any(out==p or out.is_relative_to(p) for p in roots):raise ValueError('Output must be a new directory outside all input roots')
 out.mkdir(parents=True,exist_ok=False)
 cal=Calibration.from_json(args.data/'calibration'/'calibration_20260114_114459.json')
 with args.mapping.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
 if args.partition!='all':
  wanted=set((PACKAGE/'splits'/(args.partition+'.txt')).read_text(encoding='utf-8').splitlines())
  records=[r for r in records if r['release_path'] in wanted]
 sums=defaultdict(lambda:defaultdict(float));per_frame=[]
 for rec in records:
  folder=args.data/rec['release_path'];legacy=np.load(folder/'depth_gt.npy',allow_pickle=False)
  from PIL import Image
  mask=np.asarray(Image.open(folder/'valid_mask.png'))>0
  z=corrected_rectified_z(legacy,cal) if args.convention=='corrected' else legacy.astype(float)
  d=depth_to_disparity(z,cal) if args.convention=='corrected' else np.load(folder/'disparity_gt.npy',allow_pickle=False)
  pred_relative=Path(rec['prediction_category'])/rec['prediction_session']/rec['frame']
  row={'frame':rec['release_path'],'category':rec['category'],'n_valid':int(mask.sum())}
  for method,root,filename in [('stereo',args.stereo,'disp_pred.npy'),('mono',args.mono,'depth_pred.npy')]:
   if root is None:continue
   prediction=np.load(root/pred_relative/filename,allow_pickle=False).squeeze()
   if prediction.shape!=mask.shape:raise ValueError('Prediction shape mismatch: '+rec['release_path'])
   metrics=stereo_error_sums(prediction,d,mask) if method=='stereo' else mono_error_sums(prediction,z,mask)
   for category in [rec['category'],'Overall']:
    for key,value in metrics.items():
     if key!='scale':sums[(method,category)][key]+=value
   if 'scale' in metrics:row['mono_GT_median_scale']=metrics['scale']
  per_frame.append(row)
 result=[]
 for (method,category),s in sorted(sums.items()):
  n=int(s['n']);r={'method':method,'category':category,'n_valid':n}
  if method=='stereo':r.update(EPE_px=s['sum_abs']/n,**{f'Bad{k}_pct':100*s[f'bad{k}']/n for k in [1,3,5]})
  else:r.update(AbsRel=s['sum_abs_rel']/n,RMSE_mm=math.sqrt(s['sum_sq']/n),delta1_pct=100*s['delta1']/n,delta2_pct=100*s['delta2']/n)
  result.append(r)
 payload={'status':'complete','frames':len(records),'partition':args.partition,'reference_convention':args.convention,
  'mask':'all nonzero published left-mask pixels, no display ROI','monocular_alignment':'per-frame GT median scale; predicted values then clamped to1e-3mm',
  'aggregation':'unrounded pixel sums and counts, no per-frame metric rounding','metrics':result}
 (out/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 with (out/'per_frame.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=sorted(set(k for r in per_frame for k in r)));writer.writeheader();writer.writerows(per_frame)
 print(json.dumps({'status':'complete','frames':len(records),'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
