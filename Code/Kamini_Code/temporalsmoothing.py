import numpy as np
from numpy import *
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import math
import sys


def post_proc(data,framelen,sil_thr,sp_thr):
    datasig=np.append(data,1-data[-1])
    #print datasig
    #datasig = np.logical_not(datasig).astype(int)
    sil=0
    sp=0
    framelen= float(framelen) ## in ms
    sildurthresh=np.ceil(sil_thr/framelen)
    initialsil=0
    initialsp=0
    spdurthresh=np.ceil(sp_thr/framelen)
    datasig2 = np.copy(np.flipud(datasig)) ####### Right to left
    for i in range(len(datasig)-1):######## Left to right
        if datasig[i]==0:
            sil=sil+1
        elif datasig[i]==1:
            sp=sp+1
        if datasig[i]==1 and datasig[i+1]==0:
            if sp<=spdurthresh:
                datasig[initialsp:i+1]=0    # small speech to silence
                sil=sp+sil
                sp=0
            else:
                sil=0
                finalsp=i
                initialsil=i+1
        elif datasig[i]==0 and datasig[i+1]==1:
            if sil<=sildurthresh:
                datasig[initialsil:i+1]=1   # small silence to speech
                sp=sp+sil
                sil=0
            else:
                sp=0
                finalsil=i
                initialsp=i+1
    #print datasig
    #
    for i in range(len(datasig2)-1):####### Right to left
        if datasig2[i]==0:
            sil=sil+1
        elif datasig2[i]==1:
            sp=sp+1
        if datasig2[i]==1 and datasig2[i+1]==0:
            if sp<=spdurthresh:
                datasig2[initialsp:i+1]=0    # small speech to silence
                sil=sp+sil
                sp=0
            else:
                sil=0
                finalsp=i
                initialsil=i+1
        elif datasig2[i]==0 and datasig2[i+1]==1:
            if sil<=sildurthresh:
                datasig2[initialsil:i+1]=1   # small silence to speech
                sp=sp+sil
                sil=0
            else:
                sp=0
                finalsil=i
                initialsp=i+1
    datasig = np.logical_or(datasig,np.flipud(datasig2))
    data_new=np.delete(datasig,-1)
    return data_new

# example_vad_pp = post_proc(vad_no_pp,10,200,100) # 10ms is hop size, 200ms is silence_threshold, 100ms is speech_threshold