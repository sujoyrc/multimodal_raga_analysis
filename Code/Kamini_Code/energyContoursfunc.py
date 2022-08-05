#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:32:44 2019

@author: kamini
"""

import sys
import wave
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy
import scipy.io.wavfile as wav
from scipy import signal
import pdb

def melFilter(Fs,Nfft):
    flow=0
    fhigh=Fs/2
    initmel=1125*np.log(1+flow/700)
    finalmel=1125*np.log(1+fhigh/700)
    melfreqlin=np.linspace(initmel,finalmel,22)
    melFreq=700*(np.exp(melfreqlin/1125)-1)
    melFiltBank=[]
    # print(melFreq)
    binno=np.floor((Nfft+1)*melFreq/Fs)
    for m in range(1,21):
        melwind=[]
        for k in range(Nfft):
            if k<=binno[m-1]:
                melwind.append(0)
            elif binno[m-1]<k and k<binno[m]:
                melwind.append((k-binno[m-1])/(binno[m]-binno[m-1]))
            elif k==binno[m]:
                melwind.append(1)
            elif binno[m]<k and k<binno[m+1]:
                melwind.append((k-binno[m+1])/(binno[m]-binno[m+1]))
            else:
                melwind.append(0)        
        melFiltBank.append(melwind[:Nfft//2])
    return melFiltBank


def dct2(signal):
    N=len(signal)
    dctout = np.zeros((N))
    for k in range(N):
        mult = signal*np.cos(np.pi/N*(np.array(range(N))+0.5)*k)
        dctout[k] = np.sum(mult)
    return dctout*2

def pltfontset(text_size,title_size,label_size, tick_size,legend_size,suptitle_size):
    plt.rc('font', size=text_size, weight = 'bold')          # controls default text sizes
    plt.rc('axes', titlesize=title_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=label_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=tick_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=tick_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=legend_size)    # legend fontsize
    plt.rc('figure', titlesize=suptitle_size)  # fontsize of the figure title


def displayImage(ax,matrix,figtitle,xaxislabel, yaxislabel, xaxislimit=None, yaxislimit=None):
    if xaxislimit==None:
        xaxislimit=[0,np.shape(matrix)[1]]
    if yaxislimit==None:
        yaxislimit=[0,np.shape(matrix)[0]]
    ax.imshow(matrix/np.max(matrix),extent=xaxislimit+yaxislimit,cmap='Greys',aspect='auto')
    ax.set_xlabel(xaxislabel,fontsize=14, fontweight='bold')
    ax.set_ylabel(yaxislabel,fontsize=14, fontweight='bold')
    ax.set_title(figtitle,fontsize=20, fontweight='bold')


def energyComp(axisflag,frame,silencethreshold = 0.001,window=np.array([1]),Nfft=0):
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

def energyContours(wavfilename,contourfolder):
    name=wavfilename.split('/')[-1][:-4]
    # read wav file and get info sampling rate and noSamles
    wavfile=wave.open(wavfilename,'rb')
    Fs = wavfile.getframerate()
    Ts = 1.0/Fs
    noSamples = wavfile.getnframes()

    framesizetime = 0.01
    frameSize = int(Fs*framesizetime) # noSamples
    winsizetime=0.02
    winSize = int(Fs*winsizetime) # noSamples
    Nfft = 512
    noMelCoeff = 20
    melFiltBank=melFilter(Fs,Nfft)

    # read wav file sample by sample to determine maximum absolute value of the audio signal
    maxsigampl = 0
    for i in range(noSamples):
        sample = wavfile.readframes(1)
        sampleval = struct.unpack("<h",sample)
        maxsigampl = 1.0*max(maxsigampl,np.abs(sampleval))
    wavfile.close()


    wavfile=wave.open(wavfilename,'rb')
    # initializations
    signal=[]
    frameseq=[]
    timeseq=[]
    signal1=[]
    energyContour=[]
    energy1Contour=[]
    intensityContour=[]
    intensity1Contour=[]
    band2to20EnergyContour=[]
    band2to20IntensityContour=[]
    band1Contour=[]
    band2Contour=[]
    band3Contour=[]
    band4Contour=[]
    sonoContour=[]
    sonointenContour=[]
    band1overlapContour=[]
    band2overlapContour=[]
    band3overlapContour=[]
    band4overlapContour=[]
    band1vaishaliContour=[]
    band2vaishaliContour=[]
    band3vaishaliContour=[]
    band4vaishaliContour=[]
    spectralTiltContour=[]
    spectrogram=np.empty((Nfft//2,0))
    #fid=open(contourfolder+name+'.csv','w')

    # window
    hamWin = np.hamming(winSize)

    # wav file format to read a complete frame
    fmt = "<" + "h" * frameSize

    # read wav file samples frame by frame where each frame has frameSize samples
    #framedata0 = [0]*frameSize
    #frameNo = -1 # frame counter
    extrabuffer=int(np.ceil(winSize/frameSize/2))
    maxwinSize=extrabuffer*2*frameSize # maxwinsize is always even, so we can use maxwinSize//2 safely
    frameNo = -extrabuffer
    bufferwin=[0]*maxwinSize

    # for i in range(-maxwinSize//2,noSamples,frameSize):
    for index in range(0,noSamples,frameSize):
        frame = wavfile.readframes(frameSize)
        if len(frame) != 2*frameSize:
            # print('Number of samples in the frame are less than frameSize = '+ str(frameSize))
            fmt = "<" + "h" * (len(frame)//2)
        data1 = struct.unpack(fmt,frame) # frame is read as string of bytes which is in short hex format 'h' which is 2 byte long
        data = data1/maxsigampl # scaling by max amplitude

        if len(data) == frameSize:
            framedata1 = list(data)
        else:
            framedata1 = list(data)+[0]*(frameSize-len(data)) #append zeros at the end
        frameNo+=1
        bufferwin = bufferwin[frameSize:]+framedata1
    #    print(i,frameNo,len(framedata1),len(bufferwin))
        if frameNo<0:
            continue
        
    #    windata0 = np.array(framedata0+framedata1)
        frameseq.append(frameNo)
        time=frameNo*framesizetime
        timeseq.append(time)
        windata0 = bufferwin[maxwinSize//2-winSize//2:maxwinSize//2+(winSize+1)//2]
        windata1 = windata0*hamWin

        # compute energy
        frameEnergy, frameIntensity = energyComp('time',framedata1)
        winEnergy, winIntensity = energyComp('time',windata1)
        energyContour.append(winEnergy)
        intensityContour.append(winIntensity)
       
        # compute spectrum
        spectrum = np.fft.fft(windata1, n=Nfft)
        halfspectrum = spectrum[:Nfft//2]
        magspectrum = (np.abs(np.flip(halfspectrum)))*np.sqrt(2.0/Nfft) # only for plotting spectrogram as image, don't use elsewhere
        magspectrum = np.clip(magspectrum,0.0001,None)
        spectrogram=np.hstack((spectrogram,20*np.log10(magspectrum[:,np.newaxis])))

        # compute spectral band energy 
        band1 = halfspectrum[0*Nfft//Fs:500*Nfft//Fs]
        band2 = halfspectrum[500*Nfft//Fs:1000*Nfft//Fs]
        band3 = halfspectrum[1000*Nfft//Fs:2000*Nfft//Fs]
        band4 = halfspectrum[2000*Nfft//Fs:4000*Nfft//Fs]
       
        energy, intensity = energyComp('freq',halfspectrum,0.001,trapezoidalwin(len(halfspectrum)),Nfft=Nfft)
        band1energy, band1intensity = energyComp('freq',band1,0.0005,trapezoidalwin(len(band1)),Nfft=Nfft)
        band2energy, band2intensity = energyComp('freq',band2,0.0005,trapezoidalwin(len(band2)),Nfft=Nfft)
        band3energy, band3intensity = energyComp('freq',band3,0.0005,trapezoidalwin(len(band3)),Nfft=Nfft)
        band4energy, band4intensity = energyComp('freq',band4,0.0005,trapezoidalwin(len(band4)),Nfft=Nfft)
        
        energy1Contour.append(energy)
        intensity1Contour.append(intensity)
        band1Contour.append(band1intensity)
        band2Contour.append(band2intensity)
        band3Contour.append(band3intensity)
        band4Contour.append(band4intensity)
       

        # compute sonorant band energy
        sonorantBand=halfspectrum[300*Nfft//Fs:2300*Nfft//Fs]         #0.3K-2.3K
        sonorantenergy, sonorantintensity = energyComp('freq',sonorantBand,0.0005,trapezoidalwin(len(sonorantBand)),Nfft=Nfft)
        sonoContour.append(sonorantenergy)
        sonointenContour.append(sonorantintensity)
        
        # compute energy in bark bands 2 to 20 as per rosenberg AuToBI system
        band2to20=halfspectrum[200*Nfft//Fs:6500*Nfft//Fs]         #0.3K-2.3K
        band2to20energy, band2to20intensity = energyComp('freq',band2to20,0.0005,trapezoidalwin(len(band2to20)),Nfft=Nfft)
        band2to20EnergyContour.append(band2to20energy)
        band2to20IntensityContour.append(band2to20intensity)

        # compute energy across overlapping formant bands
        overlapBand1=halfspectrum[250*Nfft//Fs:1200*Nfft//Fs]   #250-1200Hz
        overlapBand2=halfspectrum[800*Nfft//Fs:3200*Nfft//Fs]   #800-3200Hz
        overlapBand3=halfspectrum[1700*Nfft//Fs:3800*Nfft//Fs]  #1700-3800Hz
        overlapBand4=halfspectrum[3000*Nfft//Fs:4700*Nfft//Fs]  #3000-4700Hz

        overlapband1energy, overlapband1intensity = energyComp('freq',overlapBand1,0.0005,trapezoidalwin(len(overlapBand1)),Nfft=Nfft)
        overlapband2energy, overlapband2intensity = energyComp('freq',overlapBand2,0.0005,trapezoidalwin(len(overlapBand2)),Nfft=Nfft)
        overlapband3energy, overlapband3intensity = energyComp('freq',overlapBand3,0.0005,trapezoidalwin(len(overlapBand3)),Nfft=Nfft)
        overlapband4energy, overlapband4intensity = energyComp('freq',overlapBand4,0.0005,trapezoidalwin(len(overlapBand4)),Nfft=Nfft)
        
        band1overlapContour.append(overlapband1intensity)
        band2overlapContour.append(overlapband2intensity)
        band3overlapContour.append(overlapband3intensity)
        band4overlapContour.append(overlapband4intensity)
        

        # compute disjoint formant bands as per Vaishali thesis
        vaishaliBand1=halfspectrum[60*Nfft//Fs:400*Nfft//Fs]   #60-400Hz
        vaishaliBand2=halfspectrum[400*Nfft//Fs:2000*Nfft//Fs]   #400-2000Hz
        vaishaliBand3=halfspectrum[2000*Nfft//Fs:5000*Nfft//Fs]  #2000-5000Hz
        vaishaliBand4=halfspectrum[5000*Nfft//Fs:8000*Nfft//Fs]  #5000-8000Hz

        vaishaliband1energy, vaishaliband1intensity = energyComp('freq',vaishaliBand1,0.0005,trapezoidalwin(len(vaishaliBand1)),Nfft=Nfft)
        vaishaliband2energy, vaishaliband2intensity = energyComp('freq',vaishaliBand2,0.0005,trapezoidalwin(len(vaishaliBand2)),Nfft=Nfft)
        vaishaliband3energy, vaishaliband3intensity = energyComp('freq',vaishaliBand3,0.0005,trapezoidalwin(len(vaishaliBand3)),Nfft=Nfft)
        vaishaliband4energy, vaishaliband4intensity = energyComp('freq',vaishaliBand4,0.0005,trapezoidalwin(len(vaishaliBand4)),Nfft=Nfft)
        
        band1vaishaliContour.append(vaishaliband1intensity)
        band2vaishaliContour.append(vaishaliband2intensity)
        band3vaishaliContour.append(vaishaliband3intensity)
        band4vaishaliContour.append(vaishaliband4intensity)
        

        # spectral tilt using MFCC
        mellogenergy=[]
        if np.all(halfspectrum==0.0):
            spectralTilt=0.0
        else:
            for mb in range(noMelCoeff):
                melspectrum=halfspectrum*melFiltBank[mb]
                melenergy, melintensity = energyComp('freq',melspectrum,0.0005,Nfft=Nfft)
                mellogenergy.append(melintensity)
            dctlist=scipy.fft.dct(mellogenergy)
            spectralTilt=dctlist[1]
        spectralTiltContour.append(spectralTilt)

    #    framedata0 = framedata1
        signal1.append(framedata1)
        signal.extend(framedata1)
        
        
       
    #    fid.write(energyContour,intensityContour,sonoContour,sonointenContour, \
    #band1Contour,band2Contour,band3Contour,band4Contour, \
    #band1overlapContour,band2overlapContour,band3overlapContour,band4overlapContour, \
    #band1vaishaliContour,band2vaishaliContour,band3vaishaliContour,band4vaishaliContour, \
    #spectralTiltContour+'\n')
    wavfile.close()

    ## save all the contours in respective folders
    #np.savetxt(energyfolder+name+'full.txt',(energyContour,intensityContour),fmt='%7.5f')
    #np.savetxt(energyfolder+name+'sono.txt',(sonoContour),fmt='%7.5f')
    #np.savetxt(spectrumBalBandfolderchrist+name+'.txt',(band1Contour,band2Contour,band3Contour,band4Contour),fmt='%7.5f')
    #np.savetxt(spectrumBalBandfolderoverlap+name+'.txt',(band1overlapContour,band2overlapContour,band3overlapContour,band4overlapContour),fmt='%7.5f')
    #np.savetxt(spectrumBalBandfoldervaishali+name+'.txt',(band1vaishaliContour,band2vaishaliContour,band3vaishaliContour,band4vaishaliContour),fmt='%7.5f')
    #np.savetxt(spectralTiltfolder+name+'.txt',spectralTiltContour,fmt='%7.5f')
    np.savetxt(contourfolder+name+'_others.csv',list(zip(frameseq,timeseq,energyContour,intensityContour,\
    sonoContour,sonointenContour,band2to20EnergyContour,band2to20IntensityContour, \
    band1Contour,band2Contour,band3Contour,band4Contour, \
    band1overlapContour,band2overlapContour,band3overlapContour,band4overlapContour, \
    band1vaishaliContour,band2vaishaliContour,band3vaishaliContour,band4vaishaliContour, \
    spectralTiltContour)),delimiter=',',fmt='%7.5f',header="frameNo,time,energy,"+\
    "intensity,sonorantEnergy,sonorantIntensity,band2to20Energy,band2to20Intensity,band1Intensity,band2Intensity,"+\
    "band3Intensity,band4Intensity,band1overlapInten,band2overlapInten,band3overlapInten,"+\
    "band4overlapInten,band1vaishaliInten,band2vaishaliInten,band3vaishaliInten,"+\
    "band4vaishaliInten,spectralTilt")
    np.savetxt(contourfolder+name+'_spectrogram.txt',spectrogram,fmt='%7.5f')

