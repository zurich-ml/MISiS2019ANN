import os
import pickle
import numpy as np
import sys

import matplotlib.pyplot as plt

def sel_eff(tp,threshold):
    
    sig_eps = np.float(np.where(tp>threshold)[0].shape[0])/np.float(tp.shape[0])
    return sig_eps

def plot_loss_acc(model_history, validation=None):
    
    if validation:
    
        plt.subplot(1,2,1)
        plt.plot(model_history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.title('Loss on validation set')
        plt.subplot(1,2,2)
        plt.plot(model_history.history['val_acc'])
        plt.xlabel('Epoch')
        plt.title('Accuracy on validation set')
        fig = plt.gcf()
        fig.set_size_inches(15,5)
        
    else:
        
        plt.subplot(1,2,1)
        plt.plot(model_history.history['loss'])
        plt.xlabel('Epoch')
        plt.title('Loss on train set')
        plt.subplot(1,2,2)
        plt.plot(model_history.history['acc'])
        plt.xlabel('Epoch')
        plt.title('Accuracy on train set')
        fig = plt.gcf()
        fig.set_size_inches(15,5)
       