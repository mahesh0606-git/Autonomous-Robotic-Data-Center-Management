"""Illustrative temporal alignment, EKF, correlation and Bayesian decision."""
import numpy as np

def align_scalar(times, values, target):
    """Linear interpolation for continuous scalar signals only; not images/point clouds."""
    t=np.asarray(times,dtype=float); v=np.asarray(values,dtype=float)
    if len(t)!=len(v) or len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError('Timestamps must be increasing and match values')
    return np.interp(target,t,v,left=np.nan,right=np.nan)

class PlanarEKF:
    """State [x,y,heading,speed,yaw_rate]; simplified planar constant-input model."""
    def __init__(self):
        self.x=np.zeros(5); self.P=np.eye(5)
    def predict(self,dt,process_var=.01):
        if dt<=0: raise ValueError('dt must be positive')
        x,y,h,v,w=self.x
        self.x=np.array([x+v*np.cos(h)*dt,y+v*np.sin(h)*dt,h+w*dt,v,w])
        F=np.eye(5)
        F[0,2]=-v*np.sin(h)*dt;F[0,3]=np.cos(h)*dt
        F[1,2]=v*np.cos(h)*dt;F[1,3]=np.sin(h)*dt;F[2,4]=dt
        self.P=F@self.P@F.T+np.eye(5)*process_var*dt
        return self.x.copy()
    def update(self,z,H,R):
        z=np.asarray(z,dtype=float); H=np.asarray(H,dtype=float);R=np.asarray(R,dtype=float)
        y=z-H@self.x; S=H@self.P@H.T+R
        K=np.linalg.solve(S,(self.P@H.T).T).T
        self.x=self.x+K@y
        I=np.eye(5);A=I-K@H
        self.P=A@self.P@A.T+K@R@K.T  # Joseph form
        return self.x.copy()

def correlations(aligned_features):
    x=np.asarray(aligned_features,dtype=float)
    if x.ndim!=2 or x.shape[0]<2: raise ValueError('Need time-by-feature matrix')
    return np.corrcoef(x,rowvar=False)

def bayesian_decision(prior,likelihood,loss):
    """Finite-hypothesis Bayes rule with action-by-hypothesis loss matrix."""
    p=np.asarray(prior,dtype=float); l=np.asarray(likelihood,dtype=float)
    if np.any(p<0) or np.any(l<0) or np.sum(p*l)<=0: raise ValueError('Invalid probabilities')
    posterior=p*l/np.sum(p*l)
    risks=np.asarray(loss,dtype=float)@posterior
    return int(np.argmin(risks)),posterior
