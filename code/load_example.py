"""Inspect one frame and optionally export derived coordinate-converted arrays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from microdepth_tools import Calibration,load_frame,corrected_rectified_z,depth_to_disparity

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--frame',required=True)
 p.add_argument('--out',type=Path);args=p.parse_args();root=args.data.resolve();folder=(root/args.frame).resolve()
 if not folder.is_relative_to(root):raise ValueError('Frame must belong to the supplied data root')
 cal=Calibration.from_json(root/'calibration'/'calibration_20260114_114459.json');frame=load_frame(folder)
 left=corrected_rectified_z(frame['depth_original_camera_z_on_rectified_grid'],cal)
 right=corrected_rectified_z(frame['right_depth_original_camera_z_on_rectified_grid'],cal,'right')
 disp=depth_to_disparity(left,cal);mask=frame['valid_mask']
 if args.out:
  dest=args.out.resolve()
  if dest==root or dest.is_relative_to(root):raise ValueError('Derived output must be outside the input data root')
  dest.mkdir(parents=True,exist_ok=False)
  np.savez_compressed(dest/'derived_coordinates.npz',rectified_axial_z_left_mm=left,rectified_axial_z_right_mm=right,
                     rectified_disparity_px=disp,published_left_mask=mask)
 summary={'frame':args.frame,'shape':list(mask.shape),'valid_pixels':int(mask.sum()),'full_image_valid_pct':float(mask.mean()*100),
  'rectified_axial_Z_on_valid_mm':[float(left[mask].min()),float(left[mask].max())],
  'interpretation':'Coordinate convention only; floor/remap quantization, collisions and visibility are not repaired.'}
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
