"""Read-only loading and explicit geometry for the MicroDepth v1.0 release.

The published v1.0 arrays are preserved. A depth image on a rectified pixel grid
still contains original-camera axial Z. Use corrected_rectified_z before applying
the rectified stereo depth/disparity relation. These utilities do not infer
surface identity or recover the original point clouds.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Calibration:
    K_L: np.ndarray
    D_L: np.ndarray
    K_R: np.ndarray
    D_R: np.ndarray
    R_LR: np.ndarray
    T_LR: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    R_SL: np.ndarray  # p_S = R_SL p_L + t_SL; S is the calibrated texture-camera frame.
    t_SL: np.ndarray
    R_SR: np.ndarray  # p_S = R_SR p_R + t_SR.
    t_SR: np.ndarray
    baseline_mm: float
    image_size: tuple[int, int]

    @classmethod
    def from_json(cls, path: str | Path) -> 'Calibration':
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        a = data['datasetA']['calibration']; b = data['datasetB']['calibration']; ls = data['left_to_scanner']
        arr = lambda x: np.asarray(x, dtype=np.float64)
        result = cls(arr(a['K_L']),arr(a['D_L']).ravel(),arr(a['K_R']),arr(a['D_R']).ravel(),
                     arr(a['R']),arr(a['T']).reshape(3),arr(a['R1']),arr(a['R2']),arr(a['P1']),arr(a['P2']),
                     arr(ls['R']),arr(ls['T']).reshape(3),arr(b['R_RS']),arr(b['T_RS']).reshape(3),
                     float(a['baseline_mm']),tuple(data['datasetA']['image_size']))
        for matrix in [result.R_LR,result.R1,result.R2,result.R_SL,result.R_SR]:
            if matrix.shape != (3,3) or not np.allclose(matrix.T @ matrix,np.eye(3),atol=1e-7):
                raise ValueError('A calibration rotation is not orthonormal')
        if not np.isclose(-result.P2[0,3]/result.P2[0,0],result.baseline_mm,rtol=1e-7):
            raise ValueError('The horizontal rectified baseline and calibration baseline disagree')
        return result


def load_frame(folder: str | Path) -> dict:
    folder = Path(folder)
    result = {
        'left_rgb': np.asarray(Image.open(folder/'rectified_left.png').convert('RGB')),
        'right_rgb': np.asarray(Image.open(folder/'rectified_right.png').convert('RGB')),
        'depth_original_camera_z_on_rectified_grid': np.load(folder/'depth_gt.npy',allow_pickle=False),
        'right_depth_original_camera_z_on_rectified_grid': np.load(folder/'depth_gt_right.npy',allow_pickle=False),
        'legacy_disparity': np.load(folder/'disparity_gt.npy',allow_pickle=False),
        'valid_mask': np.asarray(Image.open(folder/'valid_mask.png')) > 0,
        'consistency_score': np.asarray(Image.open(folder/'confidence_rect.png')),
        'metadata': json.loads((folder/'frame_info.json').read_text(encoding='utf-8-sig')),
    }
    shape = result['valid_mask'].shape
    for name in ['depth_original_camera_z_on_rectified_grid','right_depth_original_camera_z_on_rectified_grid','legacy_disparity','consistency_score']:
        if result[name].shape != shape:
            raise ValueError(f'Inconsistent array shape: {name}')
    return result


def axial_factor(calibration: Calibration, shape: tuple[int,int], view: Literal['left','right']='left') -> np.ndarray:
    if view not in {'left','right'}: raise ValueError('view must be left or right')
    r,p = (calibration.R1,calibration.P1) if view=='left' else (calibration.R2,calibration.P2)
    v,u = np.indices(shape,dtype=np.float64)
    factor = r[0,2]*(u-p[0,2])/p[0,0] + r[1,2]*(v-p[1,2])/p[1,1] + r[2,2]
    if np.any(factor<=0) or not np.isfinite(factor).all():
        raise ValueError('Nonpositive original-camera ray Z in requested rectified image')
    return factor


def corrected_rectified_z(legacy_z_mm: np.ndarray, calibration: Calibration,
                           view: Literal['left','right']='left') -> np.ndarray:
    """Change the axial coordinate after the v1.0 inverse image remap.

    Rectification rotates the basis of each camera about its own origin; no
    translation enters this per-camera change. At integer pixel-centre (u,v),
    X_orig=R_rect.T @ (Z_rect*[x_rect,y_rect,1]). OpenCV calibration/image-grid
    coordinates are used consistently; no extra +0.5 pixel shift is added.
    Thus Z_orig=Z_rect*(R_rect[:,2] dot [x_rect,y_rect,1]).
    Zero remains zero. No hole filling or change to the released valid mask occurs.
    """
    z = np.asarray(legacy_z_mm,dtype=np.float64)
    out = np.zeros(z.shape,dtype=np.float64)
    good = np.isfinite(z) & (z>0)
    factor = axial_factor(calibration,z.shape,view)
    out[good] = z[good]/factor[good]
    return out


def depth_to_disparity(z_rect_mm: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Rectified LEFT axial Z (mm) to LEFT-minus-RIGHT disparity (pixels).

    The archived horizontal calibration has positive B=-P2[0,3]/P2[0,0],
    so P2[0,3]=-f*B<0. Thus d=f*B/Z+(cx_left-cx_right).
    """
    z = np.asarray(z_rect_mm,dtype=np.float64)
    out = np.zeros(z.shape,dtype=np.float64)
    good = np.isfinite(z) & (z>0)
    out[good] = -calibration.P2[0,3]/z[good] + calibration.P1[0,2]-calibration.P2[0,2]
    return out


def disparity_to_depth(disparity_px: np.ndarray, calibration: Calibration) -> np.ndarray:
    d = np.asarray(disparity_px,dtype=np.float64)
    denominator = d-calibration.P1[0,2]+calibration.P2[0,2]
    out = np.zeros(d.shape,dtype=np.float64)
    good = np.isfinite(denominator) & (denominator>0)
    out[good] = -calibration.P2[0,3]/denominator[good]
    return out


def native_to_reference(points_native_mm: np.ndarray, command_delta_mm: float,
                        apparent_axis_S: np.ndarray, empirical_offset_S_mm: np.ndarray,
                        rotation: Literal['CCW90','CW90']='CCW90') -> np.ndarray:
    """Explicit legacy coordinate correction; not a physical-center calibration.

    apparent_axis_S denotes observed target-coordinate motion for +1mm command.
    A static point is restored by subtracting apparent_axis_S*command_delta_mm.
    Invalid native zeros are removed BEFORE translating coordinates.
    """
    p = np.asarray(points_native_mm,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3: raise ValueError('Expected N x 3 millimeter points')
    a = np.asarray(apparent_axis_S,dtype=float).reshape(3)
    if not np.isclose(np.linalg.norm(a),1.,atol=1e-7): raise ValueError('Stage axis must be a unit vector')
    p = p[np.isfinite(p).all(axis=1)&(p[:,2]>0)]
    if rotation=='CCW90': q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    elif rotation=='CW90': q=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    else: raise ValueError('Unsupported native-coordinate rotation')
    return p@q.T + np.asarray(empirical_offset_S_mm,dtype=float).reshape(3)-a*float(command_delta_mm)


def reference_to_camera(points_S0_mm: np.ndarray, calibration: Calibration,
                        view: Literal['left','right']='left') -> np.ndarray:
    if view=='left': r,t=calibration.R_SL,calibration.t_SL
    elif view=='right': r,t=calibration.R_SR,calibration.t_SR
    else: raise ValueError('view must be left or right')
    return (np.asarray(points_S0_mm,dtype=float)-t)@r


def rasterize(points_camera_mm: np.ndarray, K: np.ndarray, D: np.ndarray,
              image_size: tuple[int,int], reducer: Literal['legacy_mean','nearest_z']='legacy_mean') -> np.ndarray:
    """Distorted-camera projection with floor binning and explicit collision rule.

    legacy_mean reproduces the audited source collision rule. nearest_z is an
    alternative for point clouds that a user possesses, not a claim about v1.0.
    Neither rule guarantees correct surface identity when foreground samples are absent.
    """
    import cv2
    p=np.asarray(points_camera_mm,dtype=np.float64)
    p=p[np.isfinite(p).all(axis=1)&(p[:,2]>0)]
    W,H=image_size
    out=np.zeros((H,W),dtype=np.float64)
    if not len(p): return out
    # Points are already in this camera frame, so zero pose applies only the
    # pinhole projection and Brown-Conrady distortion; this is not a pose estimate.
    uv=cv2.projectPoints(p.reshape(-1,1,3),np.zeros(3),np.zeros(3),np.asarray(K,dtype=float),np.asarray(D,dtype=float).ravel()[:5])[0].reshape(-1,2)
    inside=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    ij=np.floor(uv[inside]).astype(np.int64);z=p[inside,2]
    if reducer=='legacy_mean':
        count=np.zeros((H,W),dtype=np.int64)
        np.add.at(out,(ij[:,1],ij[:,0]),z);np.add.at(count,(ij[:,1],ij[:,0]),1)
        np.divide(out,count,out=out,where=count>0)
    elif reducer=='nearest_z':
        out.fill(np.inf);np.minimum.at(out,(ij[:,1],ij[:,0]),z);out[~np.isfinite(out)]=0
    else: raise ValueError('Unknown collision reducer')
    return out


def consistency_score(depth_L_mm: np.ndarray, depth_R_mm: np.ndarray, calibration: Calibration,
                       mode: Literal['legacy','legacy_no_closing','undistorted_no_closing']='legacy') -> tuple[np.ndarray,np.ndarray]:
    """LEFT score on ORIGINAL distorted-camera grids, plus raw comparable mask.

    legacy reproduces the audited pinhole-backprojection approximation and closing.
    undistorted_no_closing corrects the ray calculation and propagates no score.
    Both scores measure agreement between two derived depth maps, not true error.
    """
    import cv2
    if mode not in {'legacy','legacy_no_closing','undistorted_no_closing'}: raise ValueError('Unknown consistency mode')
    L=np.asarray(depth_L_mm);R=np.asarray(depth_R_mm)
    if L.shape!=R.shape: raise ValueError('Depth maps must share an image shape')
    H,W=L.shape;yy,xx=np.nonzero(np.isfinite(L)&(L>0));z=L[yy,xx].astype(float)
    score=np.zeros((H,W),dtype=np.uint8);comparable=np.zeros((H,W),dtype=bool)
    if not len(z): return score,comparable
    pixels=np.column_stack([xx,yy]).astype(float)
    if mode.startswith('legacy'):
        rays=(pixels-calibration.K_L[:2,2])/np.array([calibration.K_L[0,0],calibration.K_L[1,1]])
    else:
        rays=cv2.undistortPoints(pixels.reshape(-1,1,2),calibration.K_L,calibration.D_L).reshape(-1,2)
    p=np.column_stack([rays*z[:,None],z])@calibration.R_LR.T+calibration.T_LR
    front=np.isfinite(p).all(axis=1)&(p[:,2]>0);p=p[front];yy=yy[front];xx=xx[front]
    if not len(p): return score,comparable
    uv=cv2.projectPoints(p.reshape(-1,1,3),np.zeros(3),np.zeros(3),calibration.K_R,calibration.D_R)[0].reshape(-1,2)
    good=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    uv=np.floor(uv[good]).astype(int);p=p[good];yy=yy[good];xx=xx[good]
    other=R[uv[:,1],uv[:,0]].astype(float);good=np.isfinite(other)&(other>0)
    e=np.abs(p[good,2]-other[good])/((p[good,2]+other[good])/2+1e-6)
    score[yy[good],xx[good]]=np.clip(255*(1-e/0.1),0,255).astype(np.uint8)
    comparable[yy[good],xx[good]]=True
    if mode=='legacy' and score.max()>0:
        score=cv2.morphologyEx(score,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    return score,comparable


def stereo_error_sums(prediction_px: np.ndarray, reference_px: np.ndarray, valid: np.ndarray) -> dict:
    """Accumulate all requested GT-valid pixels without silent prediction filtering."""
    p=np.asarray(prediction_px,dtype=np.float64)[valid];g=np.asarray(reference_px,dtype=np.float64)[valid]
    if not len(g): return {'n':0,'sum_abs':0.,'bad1':0,'bad3':0,'bad5':0}
    if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions on the evaluation mask; report failures explicitly')
    if not np.isfinite(g).all() or np.any(g<=0): raise ValueError('Reference mask contains invalid disparity')
    error=np.abs(p-g)
    return {'n':len(g),'sum_abs':float(error.sum()),'bad1':int((error>1).sum()),'bad3':int((error>3).sum()),'bad5':int((error>5).sum())}


def mono_error_sums(prediction_relative: np.ndarray, reference_mm: np.ndarray, valid: np.ndarray) -> dict:
    p=np.asarray(prediction_relative,dtype=np.float64)[valid];g=np.asarray(reference_mm,dtype=np.float64)[valid]
    if not len(g):return {'n':0,'sum_abs_rel':0.,'sum_sq':0.,'delta1':0,'delta2':0}
    if not np.isfinite(p).all() or np.median(p)<=0: raise ValueError('Invalid monocular prediction')
    scale=np.median(g)/np.median(p)
    p=np.maximum(p*scale,1e-3);g=np.maximum(g,1e-3)
    ratio=np.maximum(p/g,g/p)
    return {'n':len(g),'sum_abs_rel':float(np.sum(np.abs(p-g)/g)),'sum_sq':float(np.sum((p-g)**2)),
            'delta1':int((ratio<1.25).sum()),'delta2':int((ratio<1.25**2).sum()),'scale':float(scale)}
