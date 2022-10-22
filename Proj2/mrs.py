#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa
import librosa.display
import sounddevice as sd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st

musics=np.array(["MT0000202045.mp3","MT0000379144.mp3","MT0000414517.mp3","MT0000956340.mp3"])

def normalize(array):
    nl, nc = array.shape
    for i in range(nc):
        vmax = array[:, i].max()
        vmin = array[:, i].min()
        array[:, i] = (array[:, i] - vmin)/(vmax - vmin)
    return array

def ex2_1():
    fileName = './datasets/features/top100_features.csv'
    top100 = np.genfromtxt(fileName, delimiter=',')
    nl, nc = top100.shape
    top100 = top100[1:, 1:(nc-1)]
    
    top100_N = np.zeros(top100.shape)
    top100_N=normalize(top100)
        
    fileName = './datasets/avp_features/top100_features_normalized_data.csv'
    np.savetxt(fileName, top100_N, fmt = "%lf", delimiter=',')
    
def auxFunc1(array):
    mean=np.mean(array)
    std =np.std(array)
    skew = st.skew(array)
    kurtosis=st.kurtosis(array)
    median=np.median(array)
    vmax = array.max()
    vmin = array.min()
    return np.array([mean,std, skew, kurtosis,median,vmax,vmin])

def ex2_2():
    numFiles = len(files)
    sampleRate = 22050
    useMono = True
    warnings.filterwarnings("ignore")
    F0_minFreq = 20 #minimun audible frequency
    F0_maxFreq = 11025 #nyquist/2
    mfcc_dim = 13
    spContrast_dim = 7 #(6+1) bands
    size=(13+14)*7+1
    res=np.zeros((numFiles, size))
    for n in range(numFiles):
        pos=0
        fileName = filesPath + files[n]
        inFile = librosa.load(fileName, sr=sampleRate, mono = useMono)[0]
        
        mfcc = librosa.feature.mfcc(inFile, n_mfcc = mfcc_dim)
        
        nl, nc = mfcc.shape
        temp = np.zeros((27, 7))
        for i in range(nl):
            temp[pos, :] = auxFunc1(mfcc[i, :])
            pos=pos+1
            
        centroid=librosa.feature.spectral_centroid(inFile)
        temp[pos,:] = auxFunc1(centroid[0,:])
        pos=pos+1
        
        bandwidth=librosa.feature.spectral_bandwidth(inFile)
        temp[pos,:] = auxFunc1(bandwidth[0,:])
        pos=pos+1
        
        contrast=librosa.feature.spectral_contrast(inFile)
        nl, nc = contrast.shape
        for i in range(nl):
            temp[pos, :] = auxFunc1(contrast[i, :])
            pos=pos+1
        
        flatness=librosa.feature.spectral_flatness(inFile)
        temp[pos,:] = auxFunc1(flatness[0,:])
        pos=pos+1
        
        rolloff=librosa.feature.spectral_rolloff(inFile)
        temp[pos,:] = auxFunc1(rolloff[0,:])
        pos=pos+1
        
        F0=librosa.yin(inFile,fmin=F0_minFreq,fmax=F0_maxFreq)
        F0[F0==F0_maxFreq]=0
        temp[pos,:] = auxFunc1(F0)
        pos=pos+1
        
        rms=librosa.feature.rms(inFile)
        temp[pos,:] = auxFunc1(rms[0,:])
        pos=pos+1
        
        crossing_rate=librosa.feature.zero_crossing_rate(inFile)
        temp[pos,:] = auxFunc1(crossing_rate[0,:])
        pos=pos+1
        
        final = temp.flatten()
        
        time=librosa.beat.tempo(inFile)
        final=np.append(final,time)
        
        res[n,:]=final
    fileName = './datasets/avp_features/all_audio_features_normalized_data.csv'
    res=normalize(res)
    np.savetxt(fileName, res, fmt = "%lf", delimiter=',')
    
def calc_vals(info_file,name):
    res=np.zeros((900,900))
    for c in range(900):
        res[c][c]=-1;
        for i in range(c+1,900):
            val =np.sqrt(np.nansum(np.square(info_file[c] - info_file[i])))
            res[c][i]=val;
            res[i][c]=val;
    saveFileName='./datasets/avp_features/'+name+'_features_Euclidiana.csv'
    np.savetxt(saveFileName, res, fmt = "%lf", delimiter=',')
    for c in range(900):
        for i in range(c+1,900):
            val=np.nansum(np.abs(info_file[c] - info_file[i]))
            res[c][i]=val;
            res[i][c]=val;
    saveFileName='./datasets/avp_features/'+name+'_features_Manhattan.csv'
    np.savetxt(saveFileName, res, fmt = "%lf", delimiter=',')
    for c in range(900):
        for i in range(c+1,900):
            val =1-(np.nansum(info_file[c]*info_file[i])/((np.sqrt(np.nansum(np.square(info_file[c]))))*(np.sqrt(np.nansum(np.square(info_file[i]))))));
            res[c][i]=val;
            res[i][c]=val;
    saveFileName='./datasets/avp_features/'+name+'_features_coseno.csv'
    np.savetxt(saveFileName, res, fmt = "%lf", delimiter=',')
    

def ex3_1():
    fileName = './datasets/avp_features/top100_features_normalized_data.csv'
    current_file = np.genfromtxt(fileName, delimiter=',')
    calc_vals(current_file,'top100')
    
    fileName = './datasets/avp_features/all_audio_features_normalized_data.csv'
    current_file = np.genfromtxt(fileName, delimiter=',')
    calc_vals(current_file,'Librosa')
            

def print_result(index,file,comp):
    if(comp==1):
        fileName='./datasets/avp_features/'+file+'_features_Euclidiana.csv'
    elif(comp==2):
        fileName='./datasets/avp_features/'+file+'_features_Manhattan.csv'
    else :
        fileName='./datasets/avp_features/'+file+'_features_coseno.csv'
    current_file = np.genfromtxt(fileName, delimiter=',')
    file_line=current_file[index]
    top20_index=np.argsort(file_line)
    top20=np.empty(20, dtype=object);
    for c in range(20):
        top20[c]=files[top20_index[c+1]]
    
    if(comp==1):
        print("Ranking: Euclidiana")
    elif(comp==2):
        print("Ranking: Manhattan")
    else:
        print("Ranking: Coseno")
        
    print(top20)
    print()

def print_music_results(i,music):
    print("----------------"+music+"-------------------")
    print()
    index = files.index(music)
    print("-----------Top 100 features------------")
    print()
    print_result(index,'top100',1)
    print_result(index,'top100',2)
    print_result(index,'top100',3)
        
    print("-----------Librosa features------------")
    print()
    print_result(index,'Librosa',1)   
    print_result(index,'Librosa',2)
    print_result(index,'Librosa',3)
    
    print()
    print()

def ex3_3():
    for i in range(4):
        print_music_results(i,musics[i])
      
def print_top20_metadata(i,name):
    print("----------------"+name+"-------------------")
    print()
    index = files.index(name)
    fileName='./datasets/avp_features/metadata_comparation.csv'
    current_file = np.genfromtxt(fileName, delimiter=',')
    file_line=current_file[index]
    top20_index=np.argsort(file_line)
    top20=np.empty(20, dtype=object);
    for c in range(20):
        top20[c]=files[top20_index[900-1-c]]
    
    print("Ranking:")
    print(top20)
    print()
    print()
    
def ex4_1_dataSet():
    metadataRawMatrix = np.genfromtxt('datasets/audio/panda_dataset_taffc_metadata.csv', delimiter=',', dtype="str")
    metadata = metadataRawMatrix[1:, [1, 3, 9, 11]]

    metadataScores = np.zeros((900, 900))
    for c in range(900):
        metadataScores[c, c] = -1
        for i in range(c+1, metadata.shape[0]):
            score = 0
            for j in range(metadata.shape[1]):
                #teste para artista e quadrante:
                if j < 2:
                    if metadata[0, j] == metadata[i, j]:
                        score = score + 1
                else:
                    #teste para MoodStrSplit e GenresStr
                    listA = metadata[c, j][1:-1].split('; ')#retira as "" do comeÃ§o e fim e separa no "; "
                    listB = metadata[i, j][1:-1].split('; ')
                    for e in listA:
                        for ee in listB:
                            if e == ee:
                                score = score + 1
            metadataScores[c, i] = score
            metadataScores[i, c] = score
            
    saveFileName='./datasets/avp_features/metadata_comparation.csv'
    np.savetxt(saveFileName, metadataScores, fmt = "%lf", delimiter=',')

def ex4_1_getTop20():
    for i in range(4):
        print_top20_metadata(i,musics[i])
       
        
def get_result(index,file,comp):
    if(comp==1):
        fileName='./datasets/avp_features/'+file+'_features_Euclidiana.csv'
    elif(comp==2):
        fileName='./datasets/avp_features/'+file+'_features_Manhattan.csv'
    else :
        fileName='./datasets/avp_features/'+file+'_features_coseno.csv'
    
    current_file = np.genfromtxt(fileName, delimiter=',')
    file_line=current_file[index]
    top20_index=np.argsort(file_line)
    top20=np.empty(20, dtype=object);
    for c in range(20):
        top20[c]=files[top20_index[c+1]]
    
    return top20
    
def get_result_metadata(index):
    fileName='./datasets/avp_features/metadata_comparation.csv'
    current_file = np.genfromtxt(fileName, delimiter=',')
    file_line=current_file[index]
    top20_index=np.argsort(file_line)
    top20=np.empty(20, dtype=object);
    for c in range(20):
        top20[c]=files[top20_index[900-1-c]]
        
    return top20
       
def calc_presision():
    indexs=np.array([0,0,0,0])
    for i in range(4):
        indexs[i] = files.index(musics[i])

    
    for i in range(3):
        value=0
        for v in range(4):
            temp=((len(np.intersect1d(get_result(indexs[v],'top100',i+1),get_result_metadata(indexs[v])))/20)*100)
            print(temp)
            value=value+temp
        if(i==0):
            print("presision de100:",value/4)
        elif(i==1):
            print("presision dm100:",value/4)
        else:
            print("presision dc100:",value/4)
            
        value=0
        for v in range(4):
            temp=((len(np.intersect1d(get_result(indexs[v],'Librosa',i+1),get_result_metadata(indexs[v])))/20)*100)
            print(temp)
            value=value+temp
        if(i==0):
            print("presision de:",value/4)
        elif(i==1):
            print("presision dm:",value/4)
        else:
            print("presision dc:",value/4)
        
if __name__ == "__main__":
    plt.close('all')
    
    global files
    filesPath = './datasets/audio/all/'
    files = os.listdir(filesPath)
    files.sort()
    
    ex2_1()
    ex2_2()
    ex3_1()
    ex3_3()
    ex4_1_dataSet()
    ex4_1_getTop20()
    calc_presision()
    
    