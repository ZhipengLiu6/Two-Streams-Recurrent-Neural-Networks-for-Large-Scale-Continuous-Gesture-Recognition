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
from scipy.linalg.misc import norm
import scipy.io as sio
import h5py
from math import *

from readTxt import readMatrixFromTxt
from GetSubfileName import *
from readHandHog import *
def Get2DimDis(a, b):
	c = [a[0] - b[0], a[1] - b[1]]
	tmp = c[0] * c[0] + c[1] * c[1]
	#print tmp
	return sqrt(tmp)

def load_IsoValidChaLearnSkePairHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None, isvalid = None):
	outputfilepath = '/home/zhipengliu/dataset/IsoGesture/vesion2/TrainValidHOGOnly/tmpValidLostHandDetectiong.txt'
	print hogfilepath
	outputstream = open(outputfilepath, 'w')
	numValidvideo = 5784
	if isvalid == 0: # test face file is differ
		allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	else:
		allidFrame, allframeX, allframeY = readTEFaceFileliu(facepath)
	
	data = []
	label = []
	videoislist = []
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
		videoislist.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		#there is only one video which can not be deteted by faceProgramm
		if allidFrame[i-1] + 1 != idFrame and i  != 0:
			outputcontent = "%05d\n" % (idFrame - 1)
			outputstream.write(outputcontent)
		#print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]
		oneHogfilename = "HOG_%05d.txt" % idFrame
		videoid = idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename		
		if  isvalid:
			numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
			if numframe == 0:	##if it is valid data , we need all data	
				print "M_%05d.avi\n" % idFrame
				theVideolabel = theVideolabel - 1
				label[theVideolabel].append(theVideolabel)
				videoislist[theVideolabel].append(videoid)
				data[theVideolabel].append([])
				countj = len(data[theVideolabel]) - 1
				tmponefeature = np.zeros((featurelength - 1))
				data[theVideolabel][countj].append(tmponefeature)		##we use zeros when data is wrong
				continue		
		else:
			numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)# train	
		if numframe == 0:
			outputcontent = "%05d\n" % idFrame
			outputstream.write(outputcontent)
			continue
		print "M_%05d.avi\n" % idFrame
		if isvalid == 1:
			theVideolabel = 0
		theVideolabel = theVideolabel - 1
		print 'Videolabel', theVideolabel
		label[theVideolabel].append(theVideolabel)
		videoislist[theVideolabel].append(videoid)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		#print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level
		#samflag = np.zeros(numframe)
		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			##sample
			"""
			if k >= 1  and detectframe == handframe[k - 1] + 1 and samflag[handframe[k - 1]] ==0 and flag[detectframe] == 0:
				samflag[detectframe] = 1
				#print 'sample frame:', detectframe
				continue
			"""
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				elif flag[detectframe] == 3:
					continue
				data[theVideolabel][countj].append(onefeature)
		actionNumFrame = len(data[theVideolabel][countj])
		print 'actionframe', actionNumFrame
		faceposition = [faceX, faceY]
		disMax = -1
		skeFeature = []
		for t in range(actionNumFrame):
			lefthandpostion = data[theVideolabel][countj][t][leftpositionbegin: rightpositionbegin]
			righthandposition = data[theVideolabel][countj][t][rightpositionbegin: lefthandhogbegin]
			leftToRightDis = Get2DimDis(lefthandpostion, righthandposition)
			leftToFace = Get2DimDis(lefthandpostion, faceposition)
			rightToFace = Get2DimDis(righthandposition, faceposition)
			skeFeature.append([leftToRightDis, leftToFace, rightToFace])
			disMax = max(disMax, leftToRightDis, leftToFace, rightToFace)
		#normalization: pair diatance / the max distance in all frames
		for t in range(actionNumFrame):
			tmp = skeFeature[t]
			normalizeskeFeature = [tmp[0] / disMax, tmp[1] / disMax, tmp[2] / disMax]
			for kk in range(lefthandhogbegin, featurelength):
				onehogvalue = data[theVideolabel][countj][t][kk]
				normalizeskeFeature.append(onehogvalue)

			data[theVideolabel][countj][t] = normalizeskeFeature
	outputstream.close()
	return data, label, videoislist
def load_IsoValidChaLearnHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	outputfilepath = '/home/zhipengliu/dataset/IsoGesture/vesion2/TrainValidHOGOnly/tmpLostHandDetectiong.txt'
	print hogfilepath
	outputstream = open(outputfilepath, 'w')
	numValidvideo = 5784
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		#there is only one video which can not be deteted by faceProgramm
		if allidFrame[i-1] + 1 != idFrame and i  != 0:
			outputcontent = "%05d\n" % (idFrame - 1)
			outputstream.write(outputcontent)
		#print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]
		oneHogfilename = "HOG_%05d.txt" % idFrame

		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			outputcontent = "%05d\n" % idFrame
			outputstream.write(outputcontent)
			continue
		print "M_%05d.avi\n" % idFrame
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		#print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level
		#samflag = np.zeros(numframe)
		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			##sample
			"""
			if k >= 1  and detectframe == handframe[k - 1] + 1 and samflag[handframe[k - 1]] ==0 and flag[detectframe] == 0:
				samflag[detectframe] = 1
				#print 'sample frame:', detectframe
				continue
			"""
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				data[theVideolabel][countj].append(onefeature)
	for i in range(nlabel):
		onelabelSampl = len(data[i])
		for j in range(onelabelSampl):
			oneVideoLen = len(data[i][j])
			for k in range(oneVideoLen):
				tmp = data[i][j][k][lefthandhogbegin:]
				data[i][j][k] = tmp;
	return data, label
def load_IsoValidChaLearnSkePair_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	outputfilepath = '/home/zhipengliu/dataset/IsoGesture/vesion2/TrainValidHOGOnly/tmpLostHandDetectiong.txt'
	print hogfilepath
	outputstream = open(outputfilepath, 'w')
	numValidvideo = 5784
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nHogDim = 324
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		#there is only one video which can not be deteted by faceProgramm
		if allidFrame[i-1] + 1 != idFrame and i  != 0:
			outputcontent = "%05d\n" % (idFrame - 1)
			outputstream.write(outputcontent)
		#print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]
		oneHogfilename = "HOG_%05d.txt" % idFrame

		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			outputcontent = "%05d\n" % idFrame
			outputstream.write(outputcontent)
			continue
		print "M_%05d.avi\n" % idFrame
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		#print "countj", countj
		#pcahandhog = dot(handhog, pcamatrix)
		pcahandhog = handhog
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level
		#samflag = np.zeros(numframe)
		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				data[theVideolabel][countj].append(onefeature)
		actionNumFrame = len(data[theVideolabel][countj])
		faceposition = [faceX, faceY]
		disMax = -1
		for t in range(actionNumFrame):
			lefthandpostion = data[theVideolabel][countj][t][leftpositionbegin: rightpositionbegin]
			righthandposition = data[theVideolabel][countj][t][rightpositionbegin: lefthandhogbegin]
			leftToRightDis = Get2DimDis(lefthandpostion, righthandposition)
			leftToFace = Get2DimDis(lefthandpostion, faceposition)
			rightToFace = Get2DimDis(righthandposition, faceposition)
			
			disMax = max(leftToRightDis, leftToFace, rightToFace)
			skeFeature = [leftToRightDis, leftToFace, rightToFace]
			data[theVideolabel][countj][t] = skeFeature
		#normalization: pair diatance / the max distance in all frames
		
		for t in range(actionNumFrame):
			tmp = data[theVideolabel][countj][t]
			normalizeskeFeature = [tmp[0] / disMax, tmp[1] / disMax, tmp[2] / disMax]
			data[theVideolabel][countj][t] = normalizeskeFeature
	outputstream.close()
	return data, label
def load_IsoValidChaLearnClosedHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	outputfilepath = '/media/zhipengliu/zhipeng/research/competetion/IsoGesture/version1/Isovalid/tmplost.txt'
	outputstream = open(outputfilepath, 'w')
	numValidvideo = 5784
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		#there is only one video which can not be deteted by faceProgramm
		if allidFrame[i-1] + 1 != idFrame and i  != 0:
			outputcontent = "%05d\n" % (idFrame - 1)
			outputstream.write(outputcontent)

		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame

		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			outputcontent = "%05d\n" % idFrame
			outputstream.write(outputcontent)
			continue
		print "M_%05d.avi\n" % idFrame
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		lefthandpostion = [faceX + 20, level]#this means the front frame's left and right hand position
		righthandposition = [faceX - 20, level]
		IsfirstLefthand = False
		IsFirstrighthand = False
		if detectframelength >= 1:
		 	if handpositionX[0] < faceX:	#right hand
		 		righthandposition[0] = handpositionX[0]
		 		righthandposition[1] = handpositionY[0]
		 	else:
		 		lefthandpostion[0] = handpositionX[0]
		 		lefthandpostion[1] = handpositionY[0]
		if handframe[0] == handframe[1]:
			if handpositionX[1] < faceX:
				righthandposition[0] = handpositionX[1]
				righthandposition[1] = handpositionY[1]
			else:
				lefthandpostion[0] = handpositionX[1]
				lefthandpostion[1] = handpositionY[1]

		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			currentposition = [handpositionX[k], handpositionY[k]]
			Toleft = Get2DimDis(lefthandpostion, currentposition)
			Toright = Get2DimDis(righthandposition, currentposition)
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if Toright < Toleft:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
						righthandposition = [handpositionX[k], handpositionY[k]]#renew
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2
						lefthandpostion = [handpositionX[k], handpositionY[k]]                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew						
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew	
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				data[theVideolabel][countj].append(onefeature)
	for i in range(nlabel):
		onelabelSampl = len(data[i])
		for j in range(onelabelSampl):
			oneVideoLen = len(data[i][j])
			for k in range(oneVideoLen):
				tmp = data[i][j][k][lefthandhogbegin:]
				data[i][j][k] = tmp;
	outputstream.close()
	return data, label
def load_IsoValidChaLearnClosedSkePairHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	outputfilepath = '/media/zhipengliu/zhipeng/research/competetion/IsoGesture/version1/Isovalid/tmplost.txt'
	outputstream = open(outputfilepath, 'w')
	numValidvideo = 5784
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		#there is only one video which can not be deteted by faceProgramm
		if allidFrame[i-1] + 1 != idFrame and i  != 0:
			outputcontent = "%05d\n" % (idFrame - 1)
			outputstream.write(outputcontent)

		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame

		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			outputcontent = "%05d\n" % idFrame
			outputstream.write(outputcontent)
			continue
		print "M_%05d.avi\n" % idFrame
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		lefthandpostion = [faceX + 20, level]#this means the front frame's left and right hand position
		righthandposition = [faceX - 20, level]
		IsfirstLefthand = False
		IsFirstrighthand = False
		if detectframelength >= 1:
		 	if handpositionX[0] < faceX:	#right hand
		 		righthandposition[0] = handpositionX[0]
		 		righthandposition[1] = handpositionY[0]
		 	else:
		 		lefthandpostion[0] = handpositionX[0]
		 		lefthandpostion[1] = handpositionY[0]
		if handframe[0] == handframe[1]:
			if handpositionX[1] < faceX:
				righthandposition[0] = handpositionX[1]
				righthandposition[1] = handpositionY[1]
			else:
				lefthandpostion[0] = handpositionX[1]
				lefthandpostion[1] = handpositionY[1]

		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			currentposition = [handpositionX[k], handpositionY[k]]
			Toleft = Get2DimDis(lefthandpostion, currentposition)
			Toright = Get2DimDis(righthandposition, currentposition)
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if Toright < Toleft:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
						righthandposition = [handpositionX[k], handpositionY[k]]#renew
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2
						lefthandpostion = [handpositionX[k], handpositionY[k]]                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew						
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew	
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				data[theVideolabel][countj].append(onefeature)
		actionNumFrame = len(data[theVideolabel][countj])
		faceposition = [faceX, faceY]
		disMax = -1
		skeFeature = []
		for t in range(actionNumFrame):
			lefthandpostion = data[theVideolabel][countj][t][leftpositionbegin: rightpositionbegin]
			righthandposition = data[theVideolabel][countj][t][rightpositionbegin: lefthandhogbegin]
			leftToRightDis = Get2DimDis(lefthandpostion, righthandposition)
			leftToFace = Get2DimDis(lefthandpostion, faceposition)
			rightToFace = Get2DimDis(righthandposition, faceposition)
			skeFeature.append([leftToRightDis, leftToFace, rightToFace])
			disMax = max(disMax, leftToRightDis, leftToFace, rightToFace)
		#normalization: pair diatance / the max distance in all frames
		for t in range(actionNumFrame):
			tmp = skeFeature[t]
			normalizeskeFeature = [tmp[0] / disMax, tmp[1] / disMax, tmp[2] / disMax]
			for kk in range(lefthandhogbegin, featurelength):
				onehogvalue = data[theVideolabel][countj][t][kk]
				normalizeskeFeature.append(onehogvalue)

			data[theVideolabel][countj][t] = normalizeskeFeature
	outputstream.close()
	return data, label
def load_IsoValidChaLearnSkePair_LowerLevel_Hog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			continue
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			if (handpositionY[k] < level  and flag[detectframe] == 0)  or ( flag[detectframe] != 0):
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				data[theVideolabel][countj].append(onefeature)
		actionNumFrame = len(data[theVideolabel][countj])
		faceposition = [faceX, faceY]
		disMax = -1
		skeFeature = []
		for t in range(actionNumFrame):
			lefthandpostion = data[theVideolabel][countj][t][leftpositionbegin: rightpositionbegin]
			righthandposition = data[theVideolabel][countj][t][rightpositionbegin: lefthandhogbegin]
			leftToRightDis = Get2DimDis(lefthandpostion, righthandposition)
			leftToFace = Get2DimDis(lefthandpostion, faceposition)
			rightToFace = Get2DimDis(righthandposition, faceposition)
			skeFeature.append([leftToRightDis, leftToFace, rightToFace])
			disMax = max(disMax, leftToRightDis, leftToFace, rightToFace)
		#normalization: pair diatance / the max distance in all frames
		for t in range(actionNumFrame):
			tmp = skeFeature[t]
			normalizeskeFeature = [tmp[0] / disMax, tmp[1] / disMax, tmp[2] / disMax]
			for kk in range(lefthandhogbegin, featurelength):
				onehogvalue = data[theVideolabel][countj][t][kk]
				normalizeskeFeature.append(onehogvalue)

			data[theVideolabel][countj][t] = normalizeskeFeature
	return data, label
def load_IsoChaLearnSkePairHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			continue
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				data[theVideolabel][countj].append(onefeature)
		actionNumFrame = len(data[theVideolabel][countj])
		faceposition = [faceX, faceY]
		disMax = -1
		skeFeature = []
		for t in range(actionNumFrame):
			lefthandpostion = data[theVideolabel][countj][t][leftpositionbegin: rightpositionbegin]
			righthandposition = data[theVideolabel][countj][t][rightpositionbegin: lefthandhogbegin]
			leftToRightDis = Get2DimDis(lefthandpostion, righthandposition)
			leftToFace = Get2DimDis(lefthandpostion, faceposition)
			rightToFace = Get2DimDis(righthandposition, faceposition)
			skeFeature.append([leftToRightDis, leftToFace, rightToFace])
			disMax = max(disMax, leftToRightDis, leftToFace, rightToFace)
		#normalization: pair diatance / the max distance in all frames
		for t in range(actionNumFrame):
			tmp = skeFeature[t]
			normalizeskeFeature = [tmp[0] / disMax, tmp[1] / disMax, tmp[2] / disMax]
			for kk in range(lefthandhogbegin, featurelength):
				onehogvalue = data[theVideolabel][countj][t][kk]
				normalizeskeFeature.append(onehogvalue)

			data[theVideolabel][countj][t] = normalizeskeFeature
	return data, label

def load_IsoChaLearnSkePair_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			continue
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nHogDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				data[theVideolabel][countj].append(onefeature)
		actionNumFrame = len(data[theVideolabel][countj])
		faceposition = [faceX, faceY]
		disMax = -1
		for t in range(actionNumFrame):
			lefthandpostion = data[theVideolabel][countj][t][leftpositionbegin: rightpositionbegin]
			righthandposition = data[theVideolabel][countj][t][rightpositionbegin: lefthandhogbegin]
			leftToRightDis = Get2DimDis(lefthandpostion, righthandposition)
			leftToFace = Get2DimDis(lefthandpostion, faceposition)
			rightToFace = Get2DimDis(righthandposition, faceposition)
			
			disMax = max(leftToRightDis, leftToFace, rightToFace)
			skeFeature = [leftToRightDis, leftToFace, rightToFace]
			data[theVideolabel][countj][t] = skeFeature
		#normalization: pair diatance / the max distance in all frames
		
		for t in range(actionNumFrame):
			tmp = data[theVideolabel][countj][t]
			normalizeskeFeature = [tmp[0] / disMax, tmp[1] / disMax, tmp[2] / disMax]
			data[theVideolabel][countj][t] = normalizeskeFeature
		
	return data, label

def load_IsoChaLearnHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	featurelength = 4 + 56 * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			continue
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + 56
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				data[theVideolabel][countj].append(onefeature)

	for i in range(nlabel):
		onelabelSampl = len(data[i])
		for j in range(onelabelSampl):
			oneVideoLen = len(data[i][j])
			for k in range(oneVideoLen):
				tmp = data[i][j][k][lefthandhogbegin:]
				data[i][j][k] = tmp;
	return data, label
def load_MostClosed_IsoChaLearnHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nDim = 81
	featurelength = 4 + nDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			continue
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0: # first frame have two hands
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		lefthandpostion = [faceX + 20, level]#this means the front frame's left and right hand position
		righthandposition = [faceX - 20, level]
		IsfirstLefthand = False
		IsFirstrighthand = False
		if detectframelength >= 1:
		 	if handpositionX[0] < faceX:	#right hand
		 		righthandposition[0] = handpositionX[0]
		 		righthandposition[1] = handpositionY[0]
		 	else:
		 		lefthandpostion[0] = handpositionX[0]
		 		lefthandpostion[1] = handpositionY[0]
		if handframe[0] == handframe[1]:
			if handpositionX[1] < faceX:
				righthandposition[0] = handpositionX[1]
				righthandposition[1] = handpositionY[1]
			else:
				lefthandpostion[0] = handpositionX[1]
				lefthandpostion[1] = handpositionY[1]

		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			currentposition = [handpositionX[k], handpositionY[k]]
			Toleft = Get2DimDis(lefthandpostion, currentposition)
			Toright = Get2DimDis(righthandposition, currentposition)
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if Toright < Toleft:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
						righthandposition = [handpositionX[k], handpositionY[k]]#renew
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2
						lefthandpostion = [handpositionX[k], handpositionY[k]]                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew						
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew	
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				data[theVideolabel][countj].append(onefeature)
	for i in range(nlabel):
		onelabelSampl = len(data[i])
		for j in range(onelabelSampl):
			oneVideoLen = len(data[i][j])
			for k in range(oneVideoLen):
				tmp = data[i][j][k][lefthandhogbegin:]
				data[i][j][k] = tmp;
	return data, label
def load_IsoChaLearnHogSke_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	featurelength = 166
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			continue
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 85
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0:
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if handpositionX[k] < faceX:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[2] + faceX 
					if handpositionX[k] < existhandX: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					existhandX = existhandfeature[0] + faceX
					if handpositionX[k] < existhandX:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
				data[theVideolabel][countj].append(onefeature)
	return data, label
def load_MostClosed_IsoChaLearnSkePairHog_data(hogfilepath = None, facepath = None, nlabel = 249, pcamatrix = None):
	allidFrame, allframeX, allframeY = readFaceFileyin(facepath)
	data = []
	label = []
	nDim = 81
	featurelength = 4 + nDim * 2
	for i in range(nlabel):
		data.append([])
		label.append([])
	for i in range(allidFrame.size):
		idFrame = allidFrame[i]
		print "idVideo = ", idFrame
		faceX = allframeX[i]
		faceY = allframeY[i]

		oneHogfilename = "HOG_%05d.txt" % idFrame
		oneHogfilepath = hogfilepath + "/" + oneHogfilename
		numframe, theVideolabel, handframe, handrectposition, handhog = readHOG(oneHogfilepath)
		if numframe == 0:
			continue
		theVideolabel = theVideolabel - 1
		label[theVideolabel].append(theVideolabel)
		data[theVideolabel].append([])
		countj = len(data[theVideolabel]) - 1
		print "countj", countj
		pcahandhog = dot(handhog, pcamatrix)
		handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
		handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
		level = 0
		height = 240
		threshold = 10
		leftpositionbegin = 0
		rightpositionbegin = 2
		lefthandhogbegin = 4
		righthandhogbegin = 4 + nDim
		detectframelength = handpositionY.size			
		if handframe[0] != 0:			#level can judgy whether the man is actioning
			level = height - threshold
		else:
			if handframe[1] == 0: # first frame have two hands
				firsthandlevel = min(handpositionY[0], handpositionY[1])
			else:
				firsthandlevel = handpositionY[0]
			if(handframe[detectframelength - 1] == handframe[detectframelength - 2]):
				lasthandlevel = min(handpositionY[detectframelength - 1], handpositionY[detectframelength - 2])
			else:
				lasthandlevel = handpositionY[detectframelength - 1]
			level = int((firsthandlevel + lasthandlevel) / 2)  - threshold
		#print "level=",level

		flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
		detectframelength = handpositionY.size    
		lefthandpostion = [faceX + 20, level]#this means the front frame's left and right hand position
		righthandposition = [faceX - 20, level]
		IsfirstLefthand = False
		IsFirstrighthand = False
		if detectframelength >= 1:
		 	if handpositionX[0] < faceX:	#right hand
		 		righthandposition[0] = handpositionX[0]
		 		righthandposition[1] = handpositionY[0]
		 	else:
		 		lefthandpostion[0] = handpositionX[0]
		 		righthandposition[1] = handpositionY[0]
		if handframe[0] == handframe[1]:
			if handpositionX[1] < faceX:
				righthandposition[0] = handpositionX[1]
				righthandposition[1] = handpositionY[1]
			else:
				lefthandpostion[0] = handpositionX[1]
				lefthandpostion[1] = handpositionY[1]

		for k in range(detectframelength):           
			onefeature = np.zeros((featurelength)) # left hand left hog righthog
			detectframe = handframe[k]
			currentposition = [handpositionX[k], handpositionY[k]]
			Toleft = Get2DimDis(lefthandpostion, currentposition)
			Toright = Get2DimDis(righthandposition, currentposition)
			if handpositionY[k] < level :
				#print "frame=",detectframe
				if flag[detectframe] == 0:
					if Toright < Toleft:   # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 1
						righthandposition = [handpositionX[k], handpositionY[k]]#renew
					else: #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						flag[detectframe] = 2
						lefthandpostion = [handpositionX[k], handpositionY[k]]                  
				elif flag[detectframe] == 1:  # right hand is exists  already
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft: # right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY
						onefeature[0] = existhandfeature[2]
						onefeature[1] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
						onefeature[righthandhogbegin: ] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew						
					else:#left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[2]
						onefeature[3] = existhandfeature[3]
						onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
						flag[detectframe]  = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				elif flag[detectframe] == 2:#left hand is exists
					existhandfeature = data[theVideolabel][countj].pop()
					if Toright < Toleft:#right hand
						onefeature[2] = handpositionX[k] - faceX
						onefeature[3] = handpositionY[k] - faceY                          
						onefeature[0] = existhandfeature[0]
						onefeature[1] = existhandfeature[1]
						onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						onefeature[righthandhogbegin :] = pcahandhog[k]
						flag[detectframe] = 3
						righthandposition = [handpositionX[k], handpositionY[k]]#renew	
					else:                                           #left hand
						onefeature[0] = handpositionX[k] - faceX
						onefeature[1] = handpositionY[k] - faceY
						onefeature[2] = existhandfeature[0]
						onefeature[3] = existhandfeature[1]
						onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[k]
						onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
						flag[detectframe] = 3
						lefthandpostion = [handpositionX[k], handpositionY[k]]
				data[theVideolabel][countj].append(onefeature)
		actionNumFrame = len(data[theVideolabel][countj])
		faceposition = [faceX, faceY]
		disMax = -1
		skeFeature = []
		for t in range(actionNumFrame):
			lefthandpostion = data[theVideolabel][countj][t][leftpositionbegin: rightpositionbegin]
			righthandposition = data[theVideolabel][countj][t][rightpositionbegin: lefthandhogbegin]
			leftToRightDis = Get2DimDis(lefthandpostion, righthandposition)
			leftToFace = Get2DimDis(lefthandpostion, faceposition)
			rightToFace = Get2DimDis(righthandposition, faceposition)
			skeFeature.append([leftToRightDis, leftToFace, rightToFace])
			disMax = max(disMax, leftToRightDis, leftToFace, rightToFace)
		#normalization: pair diatance / the max distance in all frames
		for t in range(actionNumFrame):
			tmp = skeFeature[t]
			normalizeskeFeature = [tmp[0] / disMax, tmp[1] / disMax, tmp[2] / disMax]
			for kk in range(lefthandhogbegin, featurelength):
				onehogvalue = data[theVideolabel][countj][t][kk]
				normalizeskeFeature.append(onehogvalue)

			data[theVideolabel][countj][t] = normalizeskeFeature
	return data, label
if __name__ == '__main__':
	hogfilepath = '/media/zhipengliu/zhipeng/research/competetion/IsoGesture/version2/DepthHog/train'
	#hogfilepath = '/media/zhipengliu/zhipeng/research/competetion/IsoGesture/version2/DepthHog/valid'
	facepath = '/media/zhipengliu/zhipeng/research/competetion/IsoGesture/IsoDepthTrainFaceDetect.txt'
	#facepath = '/media/zhipengliu/zhipeng/research/competetion/IsoGesture/DepthvalidFaceDetect.txt'
	nlabel = 249
	pcaeigvectpath = '/media/zhipengliu/zhipeng/research/competetion/PCACoff.mat'
	pcamatrixfile = sio.loadmat(pcaeigvectpath)
	pcamatrix = pcamatrixfile['coeff']
	pcamatrix81 = pcamatrix[:, 0: 81]
	Validdata, label = load_IsoValidChaLearnSkePair_data(hogfilepath, facepath, nlabel, pcamatrix81)
	nFeature = 4 + 324 * 2
	timestep = 10
	print "get valid test x and y:"
	validX = []
	nptemplate = np.zeros((nFeature))
	for i in range(nlabel):
	    onelabeldata = Validdata[i]
	    onelabelLength = len(onelabeldata)

	    for j in range(onelabelLength):
	        onefeature = Validdata[i][j]
	        lenFeature = len(onefeature)
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

	validX = np.array(validX)
	print validX.shape
	h5filename = '/home/zhipengliu/kerasWork/ConGCompetetion/CNN_LSTM/hdf5Feature/Depth/allSHogTrain.h5'
	h5stream = h5py.File(h5filename, 'r')
	h5stream.create_dataset('SHogdata', data = validX)
	h5stream.close()

	"""
	sumframe = 0;
	sumvideo = 0;
	maxFrame = -1
	for i in range(nlabel):

		lablelength = len(data[i])
		sumvideo = sumvideo + lablelength
		for j in range(lablelength):
			sumframe = sumframe + len(data[i][j])
			maxFrame = max(len(data[i][j]), maxFrame)
	print sumframe, sumvideo
	mymean = sumframe / sumvideo
	print 'timesteps=', mymean
	print 'max', maxFrame
	"""