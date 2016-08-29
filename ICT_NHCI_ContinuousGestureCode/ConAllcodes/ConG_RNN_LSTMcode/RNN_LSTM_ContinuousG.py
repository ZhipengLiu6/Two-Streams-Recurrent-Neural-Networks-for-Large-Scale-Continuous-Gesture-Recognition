'''
GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_bidirectional_lstm.py
'''
#coding:utf-8
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys, os
import scipy.io as sio
import scipy.io as sio
from scipy.linalg.misc import norm
from keras.preprocessing import sequence
from keras.models import Graph
from keras.models import Sequential
from keras.layers.core import Dense, Dropout,Activation,TimeDistributedDense
from keras.layers import *
from keras.layers import Merge, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.recurrent import SimpleRNN
from keras.utils import np_utils
import keras 
from math import ceil
import copy
import scipy.io as sio
import h5py
from keras.models import model_from_json
from keras.models import load_model
from keras import regularizers

from read_chaLearnhog_data import *
from read_ISO_data import *
from generate_ConG_final_sub import *
from get_current_path import *

#------------get current file-----------
current_path = cur_file_dir()
print "current_path:", current_path
#-------------file setting----------------
##test data
Depth_Te_hogfilepath = current_path  + '/TestHogFeature/Depth'
RGB_Te_hogfilepath = current_path  + '/TestHogFeature/RGB'

##test face position data
Depth_Te_facepath = current_path  + '/FacePosition/DepthFacePosition'##test's style is different from Train and valid
RGB_Te_facepath = current_path  + '/FacePosition/RGBFacePosition' 

##---------pca file--------------------------
RGBpcapath = current_path  + '/RGBPCACoff.mat'
Depthpcapath = current_path  + '/DepthPCAoff.mat'

##---------save h5 file path------------
Depth_Te_h5file = current_path  + '/h5feature/Depth_Te_h5file.h5'
RGB_Te_h5file = current_path  + '/h5feature/RGB_Te_h5file.h5'
allTest_h5file = current_path  + '/h5feature/allTest_h5file_vid.h5'

##---------test data segmentation information data patj------
seginfopath = current_path  + '/ConGTestSegInfo'

##---------model name
modelname = current_path + '/model/ConGmodel.h5'

##----------subfile setting----------------

myConGsubfile = current_path  + '/ConGsubtmp.txt'
testvideolistfile = current_path  + '/test.txt'

##---------parameter setting------------
nlabel = 249
nhogDim = 81
nDepthHogDim = 81

RGBpcamatrixfile = sio.loadmat(RGBpcapath)
RGBpcamatrix = RGBpcamatrixfile['coeff']
RGBpcamatrixRedu = RGBpcamatrix[:, 0: nhogDim]

Depthpcamatrixfile = sio.loadmat(Depthpcapath)
Depthpcamatrix = Depthpcamatrixfile['sorteigVect']
DepthpcamatrixRedu = Depthpcamatrix[:, 0 : nDepthHogDim]
##

getFinallTestFeatureFromFile = 1##including test data

getTestFeatureAsTest = 1

getAlltestDataAsTesting = 1

IsExistModel = 1

IsGetFinalSumFile = 1
#---------------------------------------------------
timestep = 10 # mean 27
nlabel = 249
numSke = 3
nFeature = numSke + nDepthHogDim * 2
out_dim = nFeature

print "timestep =", timestep
#-------------end--------------------
for i in range(2):
    if getFinallTestFeatureFromFile == 1:
        testLabel = 81
        istest = 1
        if i == 0:#Depth
            print "load ConG Depth Test data\n"
            validh5stream = h5py.File(Depth_Te_h5file, 'w')
            Validdata, videoidlist = load_ValidConTinuous_Skepair_Hog_data(Depth_Te_hogfilepath, Depth_Te_facepath, testLabel, DepthpcamatrixRedu, seginfopath)  
        else:#RGB
            print "load ConG RGB Test data\n"
            validh5stream = h5py.File(RGB_Te_h5file, 'w')
            Validdata, videoidlist = load_ValidConTinuous_Skepair_Hog_data(RGB_Te_hogfilepath, RGB_Te_facepath, testLabel, RGBpcamatrixRedu, seginfopath)

        validX = []
        nptemplate = np.zeros((nFeature))
        datalen = len(Validdata)
        lensum = 0
        for i in range(datalen):
            onefeature = Validdata[i]
            lenFeature = len(onefeature)
            lensum = lensum + lenFeature 
            feature = []
            if lenFeature < timestep:
                for k in range(lenFeature, timestep):
                    onefeature.append(nptemplate)#The video is short and zeros later
                for tmp in onefeature:
                    feature.append(tmp)
            else:
                sampleRate = float((lenFeature) / timestep)
                for k in range(timestep):
                    mapfeatue = int(round(k * sampleRate))
                    feature.append(onefeature[mapfeatue])
            feature = np.array(feature)
            validX.append(feature)
        videoidarray = np.array(videoidlist)
        validX = np.array(validX)
        print "xtrain.shape = ", validX.shape
        validh5stream.create_dataset('validX', data = validX)
        validh5stream.create_dataset('allvideoid', data = videoidarray)
        validh5stream.close() 
        print "load continuous valid data done!\n"

if getTestFeatureAsTest == 1:

    Depthteststream = h5py.File(Depth_Te_h5file, "r")
    RGBteststream = h5py.File(RGB_Te_h5file, 'r')
    RGBxtest = RGBteststream['validX'][:]
    RGBtestvideoid = RGBteststream['allvideoid'][:]
    RGBtestdic = {-1 : -1} # videoid to lineid
    nRgbxtest = RGBxtest.shape[0]
    for i in range(nRgbxtest):
        onevideoid = RGBtestvideoid[i]
        RGBtestdic[onevideoid] = i

    Depthxtest = Depthteststream['validX'][:]
    Depthtestvideoid = Depthteststream['allvideoid'][:]
    Depthtestdic = {-1 : -1} # videoid to lineid
    nDepthxtest = Depthxtest.shape[0]
    for i in range(nDepthxtest):
        onevideoid = Depthtestvideoid[i]
        Depthtestdic[onevideoid] = i

    allNumvalidNum = 300 * 10000 * 100 ## three file layers
    newRGBxtest = []
    newDepthxtest = []
    xtestVideoId = []
    for i in range(allNumvalidNum):
        if RGBtestdic.has_key(i) and Depthtestdic.has_key(i):
            RGBline = RGBtestdic[i]
            Depthline = Depthtestdic[i]
            newRGBxtest.append(RGBxtest[RGBline, :, :])
            newDepthxtest.append(Depthxtest[Depthline, :, :])
            xtestVideoId.append(i)


    newRGBxtest = np.array(newRGBxtest)
    newDepthxtest = np.array(newDepthxtest)
    xtestVideoId = np.array(xtestVideoId)
    print "RGBtest", newRGBxtest.shape
    print 'depthtest', newDepthxtest.shape
    print 'xtestVideoId', xtestVideoId.shape

    RGBteststream.close()
    Depthteststream.close()
    alltest_stream = h5py.File(allTest_h5file, 'w')
    alltest_stream.create_dataset('RGBxtest', data = newRGBxtest)
    alltest_stream.create_dataset('Depthtest', data = newDepthxtest)
    alltest_stream.create_dataset('xtestVideoId', data = xtestVideoId)
    alltest_stream.close()

    print "load data done\n"

if getAlltestDataAsTesting == 1:

    alltest_stream = h5py.File(allTest_h5file, 'r')
    RGBxtest = alltest_stream['RGBxtest'][:]
    Depthxtest = alltest_stream['Depthtest'][:]
    xtestVideoId = alltest_stream['xtestVideoId'][:]
    alltest_stream.close()


if IsExistModel == 1:
    model = load_model(modelname)
    batch_size1 = 64
    result_class = model.predict_classes([RGBxtest, Depthxtest], batch_size=batch_size1, verbose=0)
    resultDetail = model.predict([RGBxtest, Depthxtest], batch_size=batch_size1, verbose=0)
    h5name = "/h5ConGresult.h5"
    h5refile = current_path + h5name
    h5restream = h5py.File(h5refile, "w")

    h5restream.create_dataset("result_class", data = result_class)
    h5restream.create_dataset("resultDetail", data = resultDetail)
    h5restream.create_dataset('xtestVideoId', data = xtestVideoId)
    h5restream.close()    

if IsGetFinalSumFile == 1:
    Generate_final_submission_file(h5refile, seginfopath, myConGsubfile, testvideolistfile)