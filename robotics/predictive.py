"""Untrained 72-hour risk example; NO calibrated failure predictions."""
import numpy as np
import torch
from torch import nn

def statistical_features(history):
    """History (time, channels) -> [mean,std,rms,first-to-last slope] per channel."""
    x=np.asarray(history,dtype=float)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('Expected finite time-by-channel history')
    return np.concatenate([x.mean(0),x.std(0),np.sqrt((x*x).mean(0)),(x[-1]-x[0])/(len(x)-1)])

def thermal_step(temp_c,ambient_c,heat_w,capacitance_j_per_k,conductance_w_per_k,dt_s):
    """Forward-Euler lumped thermal energy balance."""
    if capacitance_j_per_k<=0 or dt_s<=0 or conductance_w_per_k<0:
        raise ValueError('Invalid physical parameters')
    return temp_c+dt_s*(heat_w-conductance_w_per_k*(temp_c-ambient_c))/capacitance_j_per_k

class FailureLSTM(nn.Module):
    """Input (batch,time,features); 3 layers x 256 hidden units per monograph."""
    def __init__(self,input_features):
        super().__init__()
        self.lstm=nn.LSTM(input_features,256,num_layers=3,batch_first=True,dropout=.2)
        self.dropout=nn.Dropout(.2)
        self.head=nn.Linear(256,1)
    def forward(self,x):
        sequence,_=self.lstm(x)
        return torch.sigmoid(self.head(self.dropout(sequence[:,-1]))).squeeze(-1)

def mc_dropout(model,x,passes=20):
    """Descriptive stochastic model-output interval; not a calibrated CI."""
    if passes<2: raise ValueError('passes >= 2 required')
    previous=model.training
    model.eval()
    for module in model.modules():
        if isinstance(module,nn.Dropout):module.train()
    try:
        with torch.no_grad():
            y=torch.stack([model(x) for _ in range(passes)])
        return y.mean(0),torch.quantile(y,.025,dim=0),torch.quantile(y,.975,dim=0)
    finally:
        model.train(previous)
