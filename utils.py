import os
import pickle
import numpy as np
import sys

def sel_eff(tp,threshold):
    
    sig_eps = np.float(np.where(tp>threshold)[0].shape[0])/np.float(tp.shape[0])
    return sig_eps