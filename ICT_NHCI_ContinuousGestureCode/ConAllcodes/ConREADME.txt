ConGD RGB_Depth_RNN_LSTM method:


---

ConGD RGB_Depth_LSTM method: hand and face detection -> hog features and skeleton pair -> RNN_LSTM



## Notes

This code was tested on Windows10 OS with VS2012 and Ubuntu 14.04 OS with Python 2.7, keras, faster-rcnn. 
Please double check the paths in code before your run it.



##Steps in detail of RGB_Depth_LSTM method:


####Step 1. Hand detection for each video.

#####Steps for hand detection:

######Step(1). Install faster-rcnn from https://github.com/rbgirshick/py-faster-rcnn. Make sure run the ./tools/demo.py successfully

######Step(2). Copy the folder ConAllcode/HandDetectioncode/py-faster-rcnn to your faster-rcnn installed path. Replace the file if it already exists.

######Step(3). Cd py-faster-rcnn, run ./tools/chalearn_con_rgb.py to detect hands in rgb videos, then run ./tools/chalearn_con_depth.py to detect hands in depth video. Each file you may change the root_path in _main_ function.
(In ICT_NHCI_ContinuousGestureCode/ConAllcodes/HandDetectioncode/py-faster-rcnn/data/faster_rcnn_models path, trained caffe models do not exist, which can be loaded from http://pan.baidu.com/s/1gfocrfl )

######The detection results will be saved in the OriginalDetectionLabel folder which is in the same level folder as 'test'.


####Step 2. Data processing, contains refine detection results and extract hog features.

#####Steps for data processing:

######Step(1): Copy the detection results folder OriginalDetectionLabel to ConAllcode/ProcessingDatacode/output/

######Step(2): Change the filePath and Trainlist variable (line 28 and 29) in ProcessingDataConG.cpp to appropriate path.

######Step(3): Run. Depth and RGB hog feature will be generated in output/HOG/Depth/ and output/HOG/RGB/, rgb and depth face info will be generated in output/RGBFacePosition and output/DepthFacePosition

####Step 3. RNN_LSTM¡¡Classifer.

#####Steps for RNN_LSTM:

######Step(1) Install keras [https://keras.io/#installation].

######Step(2) Copy the Depth and RGB hog feature files (with folder) to ConAllcode/ConG_RNN_LSTMcode/TestHogFeature/. Copy Depth and RGB face position txt file to IsoAllcode/ConG_RNN_LSTMcode/FacePosition/. Copy the output/ConGTestSegInfo to ConAllcode/ConG_RNN_LSTMcode/. (These files can be geted from Setp 2)

######Step(3) Run RNN_LSTM_ContinuousG.py and generate a final submission file named ConGsubtmp.txt in ConAllcode/ConG_RNN_LSTMcode/


