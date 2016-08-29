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
import re

from readTxt import readMatrixFromTxt
from GetSubfileName import *
from readHandHog import *
def Get2DimDis(a, b):
	c = [a[0] - b[0], a[1] - b[1]]
	tmp = c[0] * c[0] + c[1] * c[1]
	#print tmp
	return sqrt(tmp)
def load_ConTinuous_Skepair_Hog_data(hogfilepath = None, facepath = None, nlabel = 0, pcamatrix = None):
	outputfilepath = '/home/zhipengliu/dataset/IsoGesture/vesion2/TrainValidHOGOnly/TrainLostHandDetectiong1.txt'
	outputstream = open(outputfilepath, 'w')
	r = re.compile('[ \t\n\r:]+')
	data = []
	hog = []
	face = []
	handposition = []
	label = []
	videoidlist = []
	begin = 0
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	for i in range(begin + nlabel):
		data.append([])
		label.append([])
		videoidlist.append([])

	print "filepath = " + facepath
	listsubfacefilename = GetSubfileName(facepath)
	count_invalidDetect = 0
	for i in range(begin, begin + nlabel):
		#print "-----------%d------------\n" % i
		subfacefilename = listsubfacefilename[i];
		labelfacepath = facepath + "/" + subfacefilename
		listfacefilename = GetSubfileName(labelfacepath)
		nSamepleOnelabel = len(listfacefilename)

		print nSamepleOnelabel	
		for j in range(nSamepleOnelabel):			

			
			finallFacePath = labelfacepath + "/" + listfacefilename[j]
			facestream = open(finallFacePath, 'r')
			faceline = facestream.readline()
			facelinesplit = r.split(faceline)
			faceX = int(facelinesplit[0])
			faceY = int(facelinesplit[1])

			oneHogfilename = "HOG_%03d_%04d.txt" % (i + 1, j + 1) 
			oneHogfilepath = hogfilepath + "/" + oneHogfilename
			conGvideoid = (i + 1) * 10000 + (j + 1)

			print oneHogfilename
			numframe, theVideolabel, handframe, handrectposition, detect_flag, handhog = v3_readHOG(oneHogfilepath)

			if numframe == 0:
				count_invalidDetect = count_invalidDetect + 1
				outputcontent = "HOG_%03d_%04d.txt\n" % (i + 1, j + 1) 
				outputstream.write(outputcontent)
				continue
			data[i].append([])
			label[i].append(i)
			videoidlist[i].append(conGvideoid)			
			pcahandhog = dot(handhog, pcamatrix)
			theVideolabel = theVideolabel - 1
			countj = len(data[theVideolabel]) - 1
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
	print "Wrong data:", count_invalidDetect
	return data, label, videoidlist
def load_ValidConTinuous_Skepair_Hog_data(hogfilepath = None, facepath = None, nlabel = 84, pcamatrix = None, seginfopath = None):
	outputfilepath = '/home/zhipengliu/dataset/IsoGesture/vesion2/TrainValidHOGOnly/validLostHandDetectiong1.txt'
	outputstream = open(outputfilepath, 'w')
	isofilename = '/home/zhipengliu/dataset/IsoGesture/vesion2/TrainValidHOGOnly/Isolist.txt'
	isostream = open(isofilename, 'w')
	r = re.compile('[ \t\n\r:]+')
	data = []
	hog = []
	face = []
	videoidlist = []
	handposition = []
	begin = 0
	nHogDim = 81
	featurelength = 4 + nHogDim * 2
	print "facepath = " + facepath
	print "hogpath  = " + hogfilepath
	listsubfacefilename = GetSubfileName(facepath)
	count_invalidDetect = 0
	cout_video = 0
	cout_facefile = 0
	for i in range(begin, begin + nlabel):
		#print "-----------%d------------\n" % i
		subfacefilename = listsubfacefilename[i];
		labelfacepath = facepath + "/" + subfacefilename
		labelseginfo = seginfopath + "/" + subfacefilename

		listfacefilename = GetSubfileName(labelfacepath)
		listseginfofilename = GetSubfileName(labelseginfo)
		nSamepleOnelabel = len(listfacefilename)

		print "numsubfile:", nSamepleOnelabel
		
		for j in range(nSamepleOnelabel):
			cout_video = cout_video + 1
			cout_facefile = cout_facefile + 1
			oneHogfilename = "HOG_%03d_%04d.txt" % (i + 1, j + 1) 
			conGvideoid = (i + 1) * 10000 + j + 1
			oneHogfilepath = hogfilepath + "/" + oneHogfilename
			print "hogname ", oneHogfilename
			numframe, theVideolabel, handframe, handrectposition, detect_flag, handhog = v3_readHOG(oneHogfilepath)
			if numframe == 0:
				count_invalidDetect = count_invalidDetect + 1
				outputcontent = "HOG_%03d_%04d.txt" % (i + 1, j + 1) 
				print "wrong valid video num:", count_invalidDetect
				outputstream.write(outputcontent)
				continue

			pcahandhog = dot(handhog, pcamatrix)
			handpositionY = (handrectposition[:, 1] + handrectposition[:, 3]) / 2
			handpositionX = (handrectposition[:, 0] + handrectposition[:, 2]) / 2
			

			if cout_facefile > cout_video:
				cout_video = cout_video + 1
				continue
			elif cout_facefile < cout_video:
				cout_facefile = cout_facefile + 1
				continue

			finallFacePath = labelfacepath + "/" + listfacefilename[j]
			#print "facepath ", finallFacePath
			facestream = open(finallFacePath, 'r')
			faceline = facestream.readline()
			facelinesplit = r.split(faceline)
			faceX = int(facelinesplit[0])
			faceY = int(facelinesplit[1])
			#print "facex= %d, faceY= %d" % (faceX, faceY)
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
			

			flag = np.zeros(numframe) 		# indicate hand is right(1) or left hand (2),two hand(3) flag is 0 if it does not be judged 
			detectframelength = handpositionY.size    

			oneseginfofilepath = labelseginfo + '/'  + listseginfofilename[j]
			#print "seginfopath: ", oneseginfofilepath
			seginforstream = open(oneseginfofilepath)
			alllines = seginforstream.readlines()
			numline = len(alllines)
			strcontent = listseginfofilename[j] + " " + str(numline) + "\n"
			isostream.write(strcontent)
			if numline != 0:
				for k in range(numline):
					IsoVideoid = conGvideoid * 100 + k
					data.append([])
					videoidlist.append(IsoVideoid)
					lendata = len(data) - 1
					print "lendata", lendata
					oneline = alllines[k]

					onelinesplit = r.split(oneline)
					start = int(onelinesplit[0])
					end = int(onelinesplit[1])
					
					print "%d to %d" % (start, end)
					for tk in range(detectframelength):
						detectframe = handframe[tk]
						if detectframe >= start and detectframe < end:
							onefeature = np.zeros((featurelength))
							if handpositionY[tk] < level:
								if flag[detectframe] == 0:
									if handpositionX[tk] < faceX:   # right hand
										onefeature[2] = handpositionX[tk] - faceX
										onefeature[3] = handpositionY[tk] - faceY
										onefeature[righthandhogbegin: ] = pcahandhog[tk]
										flag[detectframe] = 1
									else: #left hand
										onefeature[0] = handpositionX[tk] - faceX
										onefeature[1] = handpositionY[tk] - faceY
										onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[tk]
										flag[detectframe] = 2                  
								elif flag[detectframe] == 1:  # right hand is exists  already
									existhandfeature = data[lendata].pop()
									existhandX = existhandfeature[2] + faceX 
									if handpositionX[tk] < existhandX: # right hand
										onefeature[2] = handpositionX[tk] - faceX
										onefeature[3] = handpositionY[tk] - faceY
										onefeature[0] = existhandfeature[2]
										onefeature[1] = existhandfeature[3]
										onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
										onefeature[righthandhogbegin: ] = pcahandhog[tk]
										flag[detectframe] = 3
									else:#left hand
										onefeature[0] = handpositionX[tk] - faceX
										onefeature[1] = handpositionY[tk] - faceY
										onefeature[2] = existhandfeature[2]
										onefeature[3] = existhandfeature[3]
										onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[tk]
										onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
										flag[detectframe]  = 3
								elif flag[detectframe] == 2:#left hand is exists
									existhandfeature = data[lendata].pop()
									existhandX = existhandfeature[0] + faceX
									if handpositionX[tk] < existhandX:#right hand
										onefeature[2] = handpositionX[tk] - faceX
										onefeature[3] = handpositionY[tk] - faceY                          
										onefeature[0] = existhandfeature[0]
										onefeature[1] = existhandfeature[1]
										onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
										onefeature[righthandhogbegin :] = pcahandhog[tk]
										flag[detectframe] = 3
									else:                                           #left hand
										onefeature[0] = handpositionX[tk] - faceX
										onefeature[1] = handpositionY[tk] - faceY
										onefeature[2] = existhandfeature[0]
										onefeature[3] = existhandfeature[1]
										onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[tk]
										onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
										flag[detectframe] = 3
								data[lendata].append(onefeature)


					actionNumFrame = len(data[lendata])
					print "len=", actionNumFrame
					faceposition = [faceX, faceY]
					disMax = -1
					skeFeature = []
					for t in range(actionNumFrame):
						lefthandpostion = data[lendata][t][leftpositionbegin: rightpositionbegin]
						righthandposition = data[lendata][t][rightpositionbegin: lefthandhogbegin]
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
							onehogvalue = data[lendata][t][kk]
							normalizeskeFeature.append(onehogvalue)

						data[lendata][t] = normalizeskeFeature
			else:
				print "No video segmentation!"
				k = 1
				IsoVideoid = conGvideoid * 100 + k
				data.append([])
				videoidlist.append(IsoVideoid)
				lendata = len(data) - 1
				print "lendata", lendata
				start = 1
				end = numframe - 1
				print "%d to %d" % (start, end)
				for tk in range(detectframelength):
					detectframe = handframe[tk]
					if detectframe >= start and detectframe < end:
						onefeature = np.zeros((featurelength))
						if handpositionY[tk] < level:
							if flag[detectframe] == 0:
								if handpositionX[tk] < faceX:   # right hand
									onefeature[2] = handpositionX[tk] - faceX
									onefeature[3] = handpositionY[tk] - faceY
									onefeature[righthandhogbegin: ] = pcahandhog[tk]
									flag[detectframe] = 1
								else: #left hand
									onefeature[0] = handpositionX[tk] - faceX
									onefeature[1] = handpositionY[tk] - faceY
									onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[tk]
									flag[detectframe] = 2                  
							elif flag[detectframe] == 1:  # right hand is exists  already
								existhandfeature = data[lendata].pop()
								existhandX = existhandfeature[2] + faceX 
								if handpositionX[tk] < existhandX: # right hand
									onefeature[2] = handpositionX[tk] - faceX
									onefeature[3] = handpositionY[tk] - faceY
									onefeature[0] = existhandfeature[2]
									onefeature[1] = existhandfeature[3]
									onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[righthandhogbegin : ]
									onefeature[righthandhogbegin: ] = pcahandhog[tk]
									flag[detectframe] = 3
								else:#left hand
									onefeature[0] = handpositionX[tk] - faceX
									onefeature[1] = handpositionY[tk] - faceY
									onefeature[2] = existhandfeature[2]
									onefeature[3] = existhandfeature[3]
									onefeature[lefthandhogbegin : righthandhogbegin] = pcahandhog[tk]
									onefeature[righthandhogbegin:] = existhandfeature[righthandhogbegin: ]
									flag[detectframe]  = 3
							elif flag[detectframe] == 2:#left hand is exists
								existhandfeature = data[lendata].pop()
								existhandX = existhandfeature[0] + faceX
								if handpositionX[tk] < existhandX:#right hand
									onefeature[2] = handpositionX[tk] - faceX
									onefeature[3] = handpositionY[tk] - faceY                          
									onefeature[0] = existhandfeature[0]
									onefeature[1] = existhandfeature[1]
									onefeature[lefthandhogbegin : righthandhogbegin] = existhandfeature[lefthandhogbegin: righthandhogbegin]
									onefeature[righthandhogbegin :] = pcahandhog[tk]
									flag[detectframe] = 3
								else:                                           #left hand
									onefeature[0] = handpositionX[tk] - faceX
									onefeature[1] = handpositionY[tk] - faceY
									onefeature[2] = existhandfeature[0]
									onefeature[3] = existhandfeature[1]
									onefeature[lefthandhogbegin: righthandhogbegin] = pcahandhog[tk]
									onefeature[righthandhogbegin: ] = existhandfeature[lefthandhogbegin: righthandhogbegin]
									flag[detectframe] = 3
							data[lendata].append(onefeature)


				actionNumFrame = len(data[lendata])
				print "len=", actionNumFrame
				faceposition = [faceX, faceY]
				disMax = -1
				skeFeature = []
				for t in range(actionNumFrame):
					lefthandpostion = data[lendata][t][leftpositionbegin: rightpositionbegin]
					righthandposition = data[lendata][t][rightpositionbegin: lefthandhogbegin]
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
						onehogvalue = data[lendata][t][kk]
						normalizeskeFeature.append(onehogvalue)
					data[lendata][t] = normalizeskeFeature		

	print "Wrong data:", count_invalidDetect
	return data, videoidlist
if __name__ == '__main__':
	hogfilepath = '/media/zhipengliu/zhipeng/research/competetion/ContinuousGesture/validHog'
	facepath = '/media/zhipengliu/backupNHCI/zhipengliu/dataset/Continuous Gesture/ConGValidDataFacePosition'

	seginfopath = '/media/zhipengliu/backupNHCI/zhipengliu/dataset/Continuous Gesture/ValidContinuousVideoSegInfo'
	nlabel = 84
	pcaeigvectpath = '/media/zhipengliu/backupNHCI/zhipengliu/dataset/IsoGesture/PCACoff.mat'
	h5filename = '/home/zhipengliu/kerasWork/ConGCompetetion/CNN_LSTM/hdf5Feature/hog/ConGToIso_SHfeature_nlable249_.h5'
	h5filestream = h5py.File(h5filename, 'w')
	pcamatrixfile = sio.loadmat(pcaeigvectpath)
	pcamatrix = pcamatrixfile['coeff']
	pcamatrix81 = pcamatrix[:, 0: 81]
	data = load_ValidConTinuous_Skepair_Hog_data(hogfilepath, facepath, nlabel, pcamatrix81,seginfopath)
	print len(data)