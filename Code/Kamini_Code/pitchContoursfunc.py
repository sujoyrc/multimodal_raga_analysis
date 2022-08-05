#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:33:09 2019

@author: kamini
"""
import numpy as np
import matplotlib.pyplot as plt
# import scipy.io.wavfile as wav
#import pitchusingAutocorr as pitchcode
import sys
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import pandas as pd
# import tryFunc

def strengthlagpairs(corrnorm, minlag, maxlag, lpeak, wmax, Fs,constants):
    coarselag=[]
    finelag=[]
    acfstrength=[]
    vstrength=[]
    uvstrength=[]
    i=0
    k=0
    canddict={}
    for i in range(minlag,maxlag):
        if corrnorm[i] > corrnorm[i-1] and corrnorm[i] > corrnorm[i+1]:
            canddict[k]={}
            a = (i-1)/Fs
            b = (i)/Fs
            c = (i+1)/Fs
            xmax = blbrent(a,b,c,0.00001,Fs,(constants['interp_size'] -1)//2,0,corrnorm)
            coarselag.append(b)
            finelag.append(xmax)
            lstrength =sincinterp(xmax,Fs,(constants['interp_size'] -1)//2,int(np.floor(xmax*Fs)),corrnorm)
            acfstrength.append(lstrength)
            vstrength.append(lstrength - constants['octave_cost']*np.log2(constants['pitch_floor']*xmax))
            uvstrength.append(constants['voicing_threshold'] + max(0,2-(lpeak/wmax)/(constants['silence_threshold']/(1.0+constants['voicing_threshold']))))
            k+=1
    return k, coarselag, finelag, acfstrength, vstrength, uvstrength
    
# brent's method of interpolation
def blbrent(ax, bx, cx, tol, Fs, N, st, xn):
    e = 0.0 # flag to say that the bound is not within given tolerance
    a = min(ax,cx)
    b = max(ax,cx)
    x = w = v = bx
    st = int(np.floor(x * Fs))
    fw=fv=fx = -1 * sincinterp(x,Fs,N,st,xn)
    ITMAX=100
    ZEPS = 1.0e-10
    CGOLD = 0.3819660
    for i in range(ITMAX+1):
        xm = 0.5*(a+b)
        tol1 = tol*np.abs(x)+ZEPS
        tol2 = 2.0 * tol1
        if np.abs(x-xm) <= tol2 - (b-a)/2:
            xmin = x
            fmin=fx
            return xmin
        if np.abs(e)>tol1: # construct a trial parabolic fit
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v)*q - (x - w)*r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            etemp = e
            e = d
            if np.abs(p) >= np.abs(0.5*q*etemp) or p <= q*(a-x) or p >= q*(b-x):
                if x>=xm:
                    e=a-x
                else:
                    e=b-x
                d = CGOLD*e
            else:
                d = p/q
                u = x+d
                if u-a < tol2 or b-u < tol2:
                    d=np.sign(xm-x)*np.abs(tol1)
        else:
            if x>=xm:
                e=a-x
            else:
                e=b-x
            d = CGOLD*e
        if np.abs(d) >= tol1:
            u=x+d
        else:
            u=x+np.sign(d)*np.abs(tol1)
        st = int(np.floor(u*Fs))
        fu = -1 * sincinterp(u,Fs,N,st,xn)
        if fu <= fx:
            if u>=x:
                a=x
            else:
                b=x
            v=w
            w=x
            x=u
            fv=fw
            fw=fx
            fx=fu
        else:
            if u<x:
                a=u
            else:
                b=u
            if fu<=fw or w==x:
                v=w
                w=u
                fv=fw
                fw=fu
            elif fu<=fv or v == x or v == w:
                v=u
                fv = fu
    print('too many iterations')
    return None
      
def sincinterp(t,Fs,N, st,  x):
    st = st
    y=0.0
    for i in range(-N,N+1):
        tn = -1.0*(st+i)/Fs
        if t+tn!=0:
            sincval = np.sin(np.pi*Fs*(t+tn))/(np.pi*Fs*(t+tn))
        else:
            sincval = 1.0
        y = y+x[st+i]*sincval
    return y
    
def transitionCost(F1,F2,VoicedUnvoicedCost=0.2,OctaveJumpCost=0.2):
        if F1==0 and F2==0:
            return 0
        elif F1!=0 and F2!=0:
            return OctaveJumpCost*np.log2(F1/F2)
        else:
            return VoicedUnvoicedCost
    

def readaudio(audioName):
    Fs, data = wavfile.read(audioName)
    # scale to -1.0 -- 1.0
    if data.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif data.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    data = data / (max_nb_bit + 1.0) 
    return Fs,data


def pitchContours(wavName,pitchfolder,constants):

    name=wavName.split('/')[-1][:-4]
    voicingdecision=pd.read_csv(pitchfolder+name+'_voicingPred.csv',usecols=['voicingpredsmooth'])['voicingpredsmooth']

    Fs,y=readaudio(wavName)
    wmax = np.max(y)
    Fs = 1.0 * Fs

    fsize=Fs*constants['winlen']
    window_length = int(fsize)
    tstep = np.ceil(constants['hopsize'] * Fs)
    hop = tstep / Fs
    window_hop = int(constants['hopsize']*Fs)
    
    minlag = int(np.floor(Fs / constants['pitch_ceil']))
    maxlag = int(np.ceil(Fs / constants['pitch_floor']))    

    # ratio of autocorrelation of audio and that of hamming window
    hamwin=np.hamming(window_length)
    autocorrwin = np.correlate (hamwin,hamwin,'full')
    autocorrwin = autocorrwin[len(autocorrwin-1)//2:]
    autocorrwin[autocorrwin<=0] = 0.0
    autocorrwin = autocorrwin / autocorrwin[0]
    
    # wiritng expected pitch and candidates to files
    fidpitchmasked = open(pitchfolder+ name+'_pitchMasked.txt','w')
    fidcndmasked = open(pitchfolder+ name+'_pitchMaskedCand.txt','w')
    frameNo=-1
    candidates=[]
    strengths=[]
    vflaglist=[]
    for index in range(0,len(y),window_hop):
        frameNo+=1
        if index-window_length//2<0:
            fidpitchmasked.write(str(round(hop*frameNo,2))+'\t0.00000\n')
            fidcndmasked.write(str(frameNo*tstep/Fs)+'\t1\n0.00000\t0.00000\n')
            continue
        elif index+window_length//2-1>len(y)-1:
            fidpitchmasked.write(str(round(hop*frameNo,2))+'\t0.00000\n')
            fidcndmasked.write(str(frameNo*tstep/Fs)+'\t1\n0.00000\t0.00000\n')
            continue
        else:
            sigpart=y[index-window_length//2:index+window_length//2]
        
        lpeak=np.max(sigpart)
        windowed=sigpart*hamwin
        yy=windowed
            
        autocorrsig = np.correlate(yy,yy,'full')
        autocorrsig = autocorrsig[len(autocorrsig-1)//2:]
        autocorrsig[autocorrsig<=0] = 0.0
        autocorrsig = autocorrsig / autocorrsig[0] # divide by zero is made 0
        acfpeak=np.max(autocorrsig[minlag:maxlag])

        corrnorm = np.zeros(maxlag+constants['interp_size'])
        for i in range(minlag,maxlag+constants['interp_size']):
            if autocorrwin[i]!=0:
                corrnorm[i]=autocorrsig[i]/autocorrwin[i]
            else:
                corrnorm[i]=0.01

        # candidates computation
        num, coarselag, finelag, acfstrength, vstrength, uvstrength = strengthlagpairs(corrnorm, minlag, maxlag, lpeak, wmax, Fs,constants)
        # selecting the best candidate
        sortindex=np.argsort(vstrength)[::-1]
        coarselag=np.round(np.array(coarselag)[sortindex],6)
        finelag=np.round(np.array(finelag)[sortindex],6)
        acfstrength=np.round(np.array(acfstrength)[sortindex],6)
        vstrength=np.round(np.array(vstrength)[sortindex],6)
        uvstrength=np.round(np.array(uvstrength)[sortindex],6)       

        # voicing detection using voicing threshold and silence threshold
        if acfpeak <constants['voicing_threshold'] or lpeak < constants['silence_threshold']*wmax:
            vflag=0
        else:
            vflag=1
        vflaglist.append(vflag)
        
        num = min(num, constants['MaximumNumberOfCandidatesPerFrame'])
        cand={}
        cand['coarselag']=coarselag[:num]
        cand['finelag']=finelag[:num]
        cand['acfstrength']=acfstrength[:num]
        cand['vstrength']=vstrength[:num]
        cand['uvstrength']=uvstrength[:num]
        
        strengthlist=cand['vstrength'].tolist()#+[cand['uvstrength'][0]]*(MaximumNumberOfCandidatesPerFrame+1-num)
        pitchlist=(1.0/cand['finelag']).tolist()#+[0]*(MaximumNumberOfCandidatesPerFrame+1-num)

        if len(strengthlist)!=0 and (vflag!=0 or int(voicingdecision[frameNo])!=0):
            fidcndmasked.write(str(frameNo*tstep/Fs)+'\t'+str(num)+'\n')
            for j in range(num):
                meascost = strengthlist[j]/strengthlist[0]
                fidcndmasked.write(str(pitchlist[j])+'\t'+str(meascost)+'\n')
            fidpitchmasked.write(str(round(hop*frameNo,2))+'\t'+str(round(1.0/finelag[0],5))+'\n')
        else:
            fidcndmasked.write(str(frameNo*tstep/Fs)+'\t1\n0.00000\t0.00000\n')
            fidpitchmasked.write(str(round(hop*frameNo,2))+'\t0.00000\n')

        candidates.append(pitchlist)
        strengths.append(strengthlist)

    fidpitchmasked.close()
    fidcndmasked.close()
    noFrames=len(candidates)

    # dynamic programming based smoothing
    import os
    os.system('./dp_pda '+pitchfolder+name+'_pitchMasked.txt '+pitchfolder+name+'_pitchMaskedCand.txt '+pitchfolder+name+'_pitch.txt 1 2 0.1 2.8 0.1 650')


if __name__ == '__main__':  
    wavName=sys.argv[1] #'/home/kamini/Children_Speech_Prosody/OtherCodes/SpeechRate/CSsentAudios/test/sap_15122016_1_cs_a1_1_1.wav'#sys.argv[1]
    #pitchName='/home/kamini/Desktop/temp.txt'#sys.argv[2]
    # featurefolder=sys.argv[2] #'/home/laya/Children_Speech_Prosody/Dataset/FinalChunks/'
    pitchfolder=sys.argv[2] #featurefolder+'contourFeatures/'
    pitchContours(wavName,pitchfolder)
    