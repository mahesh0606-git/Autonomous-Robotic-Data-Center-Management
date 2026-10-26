"""Offline geometric checks only; no trajectory generation or actuation."""
from dataclasses import dataclass
import math
import numpy as np

def stereo_depth(focal_pixels,baseline_m,disparity_pixels):
    if focal_pixels<=0 or baseline_m<=0 or disparity_pixels<=0:
        raise ValueError('Stereo calibration/disparity must be positive')
    return focal_pixels*baseline_m/disparity_pixels

def position_error_mm(measured_xyz_mm,target_xyz_mm):
    a=np.asarray(measured_xyz_mm,dtype=float);b=np.asarray(target_xyz_mm,dtype=float)
    if a.shape!=(3,) or b.shape!=(3,):raise ValueError('Expected xyz')
    return float(np.linalg.norm(a-b))

@dataclass(frozen=True)
class InsertionEvidence:
    position_error_mm: float
    compensated_contact_force_n: float
    seated_depth_verified: bool
    mechanical_engagement_verified: bool
    supported_load_verified: bool

def verify_insertion(evidence,tolerance_mm=.05,contact_limit_n=5.0):
    """Conservative acceptance predicate; values are proposed, not validated."""
    return (evidence.position_error_mm<=tolerance_mm
            and 0<=evidence.compensated_contact_force_n<=contact_limit_n
            and evidence.seated_depth_verified and evidence.mechanical_engagement_verified
            and evidence.supported_load_verified)
