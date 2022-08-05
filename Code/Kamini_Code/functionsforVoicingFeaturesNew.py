# import json
import numpy as np
import pandas as pd
import scipy
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
from scipy import signal

def readaudio(audioName):
    Fs, data = wavfile.read(audioName)
    # scale to -1.0 -- 1.0
    if data.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif data.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    data = data / (max_nb_bit + 1.0) 
    return data,Fs

def autocorrfunc(**args):
    pddf=args['winseg']
    autocorr=np.sum(pddf[1:]*pddf[0:-1])/np.sqrt(np.sum(pddf[1:]**2)*np.sum(pddf[0:-1]**2))
    return autocorr

def zerocrossfunc(**args):
    pddf=args['winseg']
    pddfsign=pddf/abs(pddf)
    pddfsigndiff=np.diff(pddfsign)
    zerocross=np.logical_or(pddfsigndiff==-2, pddfsigndiff==2).sum()
    return zerocross

def energyComp(**args):
    pddf=args['winseg']
    silencethreshold = 10**(-8)
    energy = np.sum((np.abs(pddf))**2)
    intensity = 10*np.log10(energy+silencethreshold)
    return intensity    

def LPCfunc(**args):
    pddf=args['winseg']
    #LP based formant band estimation
    M = 12#Assume three formants and no noise
    # compute Mth-order autocorrelation function:
    E=np.zeros(M+1) # energy of the prediction error
    # prediction gain
    r=np.zeros(M+1) # autocorr coeff
    k=np.zeros(M+1) #reflection coeff
    a=np.zeros((M+1,M+1)) #filter coeff
    autocorr_speech=np.correlate(pddf,pddf,'full') #acf
    r=autocorr_speech[len(autocorr_speech)//2:]
    E[0]=r[0]
    for i in range(1,M+1):
        s=0
        for j in range(1,i):
            s+=r[i-j]*a[i-1,j]
        q=r[i]+s
        k[i]=-q/E[i-1]
        # print(q,k[i],E[i-1])
        E[i]=E[i-1]*(1-k[i]**2)
        for j in range(1,i):
            a[i,j]=a[i-1,j]+k[i]*a[i-1,i-j]
        a[i,i]=k[i]
    lpc=a[M,1:]
    den1=[1]+list(lpc)
    G=np.sqrt(r[0] - np.sum(np.array(-lpc) * np.array(r[1:M+1])))
    num=[G]
    residual=signal.lfilter(den1,num,pddf)#np.array(pddf)) #r[0]-np.sum(lpc*r[1:])
    Emin=r[0]-np.sum(lpc*r[1:M+1])

    # print(Emin, E[0])
    errorinten=10.0*np.log10(10**(-8)+E[0])-10.0*np.log10(10**(-6)+Emin)

    return lpc[0],errorinten

def energyCompframe(axisflag,frame,silencethreshold = 0.001,window=np.array([1]),Nfft=0):
    if len(window)==1:
        window = np.array([1]*len(frame))
    squaredSum = np.sum((np.abs(frame)*window)**2)
    if axisflag=='time':
        energy = squaredSum
    elif axisflag=='freq':
        if Nfft==0:
            print('Specify Nfft')
            return None
        energy = squaredSum/Nfft*2
    if energy < silencethreshold:
        energy = silencethreshold
        intensity = 10*np.log10(silencethreshold)
    else:
        intensity = 10*np.log10(energy)
    return energy, intensity

def trapezoidalwin(N):
    if N>=6:
        return np.array([0.25,0.5,0.75]+[1]*(N-6)+[0.75,0.5,0.25])
    else:
        print('Window length not sufficient')
        return 0

def bandenergy(**args):
    pddf=args['winseg']
    Nfft=args['args']['Nfft']
    Fs=args['args']['Fs']
    spectrum = np.fft.fft(pddf, n=Nfft)
    halfspectrum = spectrum[:Nfft//2]
    signalBand=halfspectrum[args['args']['startfreq']*Nfft//Fs:args['args']['endfreq']*Nfft//Fs]         #0.3K-2.3K
    sonorantenergy, sonorantintensity = energyCompframe('freq',signalBand,0.0005,trapezoidalwin(len(signalBand)),Nfft=Nfft)
    return sonorantenergy,sonorantintensity

def HNR(**args):
    # print(args)
    pddf=args['winseg']
    pitch_ceil=args['args']['pitch_ceil']
    pitch_floor=args['args']['pitch_floor']
    Fs=args['args']['Fs']
    autocorr = np.correlate(pddf,pddf,mode='full')
    autocorr=autocorr/np.max(autocorr)
    minlag = int(np.floor(Fs / pitch_ceil))
    maxlag = int(np.ceil(Fs / pitch_floor)) 
    # print(len(autocorr),minlag,maxlag)
    rmax=np.max(autocorr[len(autocorr-1)//2:][minlag:maxlag])
    if rmax>0.4:
        return 10*np.log10(rmax/(1-rmax))
    else:
        return -200

def contourCompute(inputdf,winSize,func,center=True,hopSize=1,**args):
    count=0
    featCon=[]
    startNan=0
    endNan=0
    if center==True:
        for index in range(0,len(inputdf),hopSize):
            count+=1
            if index-winSize//2<0:
                startNan+=1
                continue
            elif index+winSize//2-1>len(inputdf)-1:
                endNan+=1
                continue
            winseg=inputdf[index-winSize//2:index+(winSize+1)//2]
            featval=func(winseg=winseg,args=args)
            featCon.append(featval)
        try:
            featCon=startNan*[tuple([np.nan]*len(featval))]+featCon+endNan*[tuple([np.nan]*len(featval))]
        except:
            featCon=startNan*[np.nan]+featCon+endNan*[np.nan]
    else:
        for index in range(0,len(inputdf),hopSize):
            winseg=inputdf[index:index+winSize]
            featval=func(winseg=winseg,args=args)
            featCon.append(featval)
    return featCon

def voicingfeaturesCompute(audioName,winSize,hopSize):
    # voicing
    data,Fs=readaudio(audioName)
    winSize=int(winSize*Fs) #20 ms
    hopSize=int(hopSize*Fs) #10 ms
    ACFCon=contourCompute(data,winSize,autocorrfunc,hopSize=hopSize)
    ZCRCon=contourCompute(data,winSize,zerocrossfunc,hopSize=hopSize)
    IntenCon=contourCompute(data,winSize,energyComp,hopSize=hopSize)
    aCon=contourCompute(data,winSize,LPCfunc,hopSize=hopSize)
    LPCCon,LPCErrCon=np.transpose(aCon)[0],np.transpose(aCon)[1]
    HNRCon=contourCompute(data,winSize,HNR,hopSize=hopSize,pitch_floor=100,pitch_ceil=650,Fs=16000)
    smoothHNR=np.array(pd.Series(HNRCon).rolling(5,0,center=True).mean())
    sonorantBandEnergyandIntenCon=contourCompute(data,winSize,bandenergy,hopSize=hopSize,startfreq=300,endfreq=2300,Fs=16000,Nfft=512)
    sonorantBandIntenCon=np.transpose(sonorantBandEnergyandIntenCon)[1]
    nosonoBandEnergyandIntenCon=contourCompute(data,winSize,bandenergy,hopSize=hopSize,startfreq=3000,endfreq=7000,Fs=16000,Nfft=512)
    nonsonoBandIntenCon=np.transpose(nosonoBandEnergyandIntenCon)[1]
    RelBandIntenCon=sonorantBandIntenCon-nonsonoBandIntenCon
    timeaxis=np.array(range(0,len(ZCRCon)))*0.01
    # print(len(timeaxis),len(ZCRCon),len(ACFCon),len(IntenCon),len(LPCCon),len(LPCErrCon))
    contours=pd.DataFrame({'time':timeaxis,'zeroCross':ZCRCon,'intensity':IntenCon,'autoCorr':ACFCon,'LPC1':LPCCon,'error':LPCErrCon,'bandRatio':RelBandIntenCon,'HNR':smoothHNR})
    return contours




def attachnewfeatures(audioName,winSize,hopSize):
    data,Fs=readaudio(audioName)
    winSize=int(winSize*Fs) #20 ms
    hopSize=int(hopSize*Fs) #10 ms
    HNRCon=contourCompute(data,winSize,HNR,hopSize=hopSize,pitch_floor=100,pitch_ceil=650,Fs=16000)
    smoothHNR=np.array(pd.Series(HNRCon).rolling(5,0,center=True).mean())
    sonorantBandEnergyandIntenCon=contourCompute(data,winSize,bandenergy,hopSize=hopSize,startfreq=300,endfreq=2300,Fs=16000,Nfft=512)
    sonorantBandIntenCon=np.transpose(BandEnergyandIntenCon)[1]
    nosonoBandEnergyandIntenCon=contourCompute(data,winSize,bandenergy,hopSize=hopSize,startfreq=3000,endfreq=7000,Fs=16000,Nfft=512)
    nonsonoBandIntenCon=np.transpose(BandEnergyandIntenCon)[1]
    RelBandIntenCon=sonorantBandIntenCon-nonsonoBandIntenCon
    contours=pd.read_csv(featurefolder+name+'_voicing.csv',index_label='centerFrame')
    timeaxis=contours.time.values
    dataNew=pd.DataFrame({'time':timeaxis,'bandRatio':RelBandIntenCon,'HNR':smoothHNR})
    contours.merge(dataNew,on='time')
    contours.to_csv(featurefolder+name+'_voicing.csv',index_label='centerFrame')