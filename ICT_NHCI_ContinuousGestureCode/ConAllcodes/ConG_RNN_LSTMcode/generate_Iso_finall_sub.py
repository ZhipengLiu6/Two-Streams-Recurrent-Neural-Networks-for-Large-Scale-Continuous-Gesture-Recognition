# -*- coding: cp936 -*-
#funtion:read ChaLearn dataset
#author@zhipeng.liu
#date:2016.7.29
#input:Txtpath
#output:list:data and label
import sys,os
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy import *
from readTxt import readMatrixFromTxt
from GetSubfileName import *
from scipy.linalg.misc import norm
import scipy.io as sio
from readHandHog import *
import h5py
import re
from math import *
def IsInlist(onelist, key):
	for element in onelist:
		if element == key:
			return 1
	return 0
def Generate_Isofinal_submission_file(resulth5filename = None, myConGsubfile = None, testvideolistfile = None):
	r = re.compile('[ \t\n\r:,/]+')

	h5stream = h5py.File(resulth5filename, 'r')
	videoid = h5stream['xtestVideoId'][:]
	result_class = h5stream['result_class'][:]
	h5stream.close()

	NumAviVideo = videoid.shape[0]
	resultvideoDic = {-1: -1}			#videoid to result lineid
	for i in range(NumAviVideo):
		onevideoId = videoid[i]
		resultvideoDic[onevideoId] = i

	test_stream = open(testvideolistfile, 'r')
	testAllline = test_stream.readlines()
	test_stream.close()

	substream = open(myConGsubfile, 'w')
	numlen = len(testAllline)
	for i in range(numlen):
		oritestline = testAllline[i]
		onevideoId = int(oritestline[11: 16])
		linelen = len(oritestline)
		if resultvideoDic.has_key(onevideoId):
			lineid = resultvideoDic[onevideoId]
			lable = result_class[lineid] + 1
		else:
			lable = 1
		newtestline = oritestline[0 : linelen-2] + " " + str(lable) + "\n"
		substream.write(newtestline)
	substream.close()
	
if __name__ == '__main__':
	resulth5filename = '/home/zhipengliu/ChaLearn2016/experimentResult/Iso/result_isoValidhogDim_81n0V3_Depth_RGB_SkeHandHogTest_SimpleRNN_LSTM_ regular000_000_00_l2(0.01)T10_id.h5'
	myConGsubfile = '/home/zhipengliu/ChaLearn2016/IsoResult/Isosub.txt'
	testvideolistfile = '/home/zhipengliu/ChaLearn2016/IsoResult/test_list.txt'
	Generate_Isofinal_submission_file(resulth5filename, myConGsubfile, testvideolistfile)


	