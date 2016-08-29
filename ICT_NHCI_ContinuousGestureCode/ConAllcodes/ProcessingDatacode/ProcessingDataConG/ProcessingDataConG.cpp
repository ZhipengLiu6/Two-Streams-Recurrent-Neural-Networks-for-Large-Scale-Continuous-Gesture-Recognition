// ProcessingDataConG.cpp : 定义控制台应用程序的入口点。
//
#pragma once
#include "stdafx.h"
#include "PreProcessGestureData.h"
#include <fstream>
#include <string>
#include <direct.h>
#include <iostream>
#include <io.h>
#include "afx.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <math.h>
#include <stdio.h>
using namespace std;
bool FCvtVideoToImage(CString vFileNmae, vector<IplImage *>& vectorImage);
void getFiles(string path, vector<string>& files );
void GetMaxConnectedDomain(IplImage *src, IplImage *dst);
bool ExtractHogFeature(CString colorPath, CString LabelPath,int label, CString savePath);
bool ExtractHogFeature_flag(CString colorPath, CString LabelPath,int label, CString savePath);
int getCurrentDir(string path);
void getCurrentDir(string path, vector<string>& files);
void getDefinedFiles(string path, vector<string>& results);

//string filePath = "C:\\Work\\Data\\Chalearn\\IsoGD_phase_2\\IsoGD_phase_2\\test";
//CString Trainlist = "C:\\Work\\Data\\Chalearn\\IsoGD_phase_2\\IsoGD_phase_2\\test_list.txt";
//CString LabelHandFile = "C:\\Work\\Data\\Chalearn\\NeatedDetectionLabel\\IsoGD\\test";

CString ValidPath = "D:\\BaiduYunDownload\\ChaLearn\\Con_phase2\\test";
CString HandLabelPath = "..\\output\\NeatedDetectionLabel";

//#define hard_re
#define get_foreground_info
#define RefineLabel
#define getFinalLabel
#define getTestFacePosition
//#define testOneVideo
//#define CvtVideoToImage
//#define getFrontRGBvideo
#define ConGSegmentation
//#define CvtConGVideoToIsoVideo
//#define CvtRGBhandLabelToDepthLabel
void cvtRGBHandLabel2Depth(CString handLabelFilePath, CString LabelSavePath);
//#define visualizationHandDetection
//#define GetContinuousGestureHog
#define GetContinuousTestGesture
//#define GetIsoFacePosition
//#define GetIsoGestureHog
#define cvtRGBface2Depthface
//global variable
CvMat *mx1;
CvMat *my1;
CvMat *mx2;
CvMat *my2;
void fgenerateMap();
PreProcessGestureData cvtConToIsoPreProcessFace;
CString faceTxtFilename = "D:\\face.txt";

struct TwoDimension
{
	int x;
	int y;
};
struct HandBox
{
	TwoDimension A;
	TwoDimension B;
};

struct CandidateArea
{
	int tlx;
	int tly;
	int brx;
	int bry;
	int flag;
};

struct Foreground
{
	string frameID;
	int tlx;
	int tly;
	int brx;
	int bry;
};

double dis(CandidateArea rect1, CandidateArea rect2)
{
	return abs(rect1.tlx-rect2.tlx)+abs(rect1.tly-rect2.tly)+abs(rect1.brx-rect2.brx)+abs(rect1.bry-rect2.bry);
}

bool isForeground(Foreground fore, CandidateArea rect)
{
	int centerx = (rect.tlx+rect.brx)/2;
	int centery = (rect.tly+rect.bry)/2;
	if(centerx>=fore.tlx && centerx<=fore.brx && centery>=fore.tly && centery<=fore.bry)
	{
		return true;
	}
	return false;
}

double centerDis(CandidateArea rect1, CandidateArea rect2)
{
	int center1x = (rect1.tlx+rect1.brx)/2;
	int center1y = (rect1.tly+rect1.bry)/2;
	int center2x = (rect2.tlx+rect2.brx)/2;
	int center2y = (rect2.tly+rect2.bry)/2;
	return abs(center1x-center2x)+abs(center1y-center2y);
}

struct IsoVideoInfo
{
	int start;
	int end;
};
struct frameHandPosition
{
	int nframe;
	bool isDetectOneHand;//Detecting one hand is true ,else is false
	TwoDimension handA1;
	TwoDimension handA2;
	TwoDimension handB1;
	TwoDimension handB2;
};
double getTwoDimDis(TwoDimension A, TwoDimension B)
{
	return (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y) * 1.0;
}
double CvtConvideoToIsovideo(CString handLabelFilePath, CString rgbFilePath, CString IsoVideoSaveName, vector<IsoVideoInfo> & cvtIsovideoInfo, CString SegInfoDevelpath);
double CvtConvideoToIsovideo_flag(CString handLabelFilePath, CString rgbFilePath, CString IsoVideoSaveName, vector<IsoVideoInfo>& cvtIsovideoInfo, CString SegInfoDevelpath);
double fvisualHandDetection(CString handLabelFilePath, CString rgbFilePath, CString IsoVideoSaveName);
void RGB2DepthRectify(TwoDimension src, TwoDimension & dst);
int getTxtFileNum(vector<string> files)
{
	int vsize = files.size();
	int txtnum = 0;
	CString txtname = "txt";
	for(int i = 0; i < vsize; i++)
	{
		CString filename = files[i].c_str();
		CString key = filename.Right(3);
		if (key == txtname)
		{
			txtnum ++;
		}
	}
	return txtnum;
}
bool IsCorrespondingFile(CString videofile, CString labelfile)
{
	CString CSvideoid = videofile.Right(10).Left(4);
	CString CSlabelid = labelfile.Right(10).Left(4);
	if(CSvideoid == CSlabelid)
		return true;
	else
		return false;
}

void swapInt(int &a, int &b)
{
	int tmp;
	tmp = a;
	a = b;
	b = tmp;
}

int min(int a, int b)
{
	if(a<=b) return a;
	return b;
}

double gethandPosition(frameHandPosition hand, int frame);

int main()
{
	/*
	fgenerateMap();//create memory
	CString RGBface = "E:\\zhipengliu\\dataset\\IsoGesture\\IsoRGBtestFacePosition.txt";
	CString Depthface = "E:\\zhipengliu\\dataset\\IsoGesture\\IsoDepthtestFacePosition.txt";
	int videoLabel = -1;
	ofstream writestream;
	fstream trainfilestream;
	writestream.open(Depthface, ios::out|ios::trunc);//write
	trainfilestream.open(RGBface, ios::in);
	char tbuffer[256];
	int id, frame, x0, y0, x1, y1;
	TwoDimension A, B;
	while(trainfilestream.getline(tbuffer, 256))
	{
		//sscanf(tbuffer, "%s %s %d", rgbname, depthname, &label);
		sscanf(tbuffer, "%d %d %d %d",&id, &frame, &x0, &y0);
		A.x = x0;
		A.y = y0;
		RGB2DepthRectify(A, A);
		char tmp[100];
		sprintf(tmp, "%d %d %d %d\n", id, frame, A.x, A.y);
		writestream<<tmp;
	}
	writestream.close();
	trainfilestream.close();
	cvReleaseMat(&mx1);
	cvReleaseMat(&my1);
	cvReleaseMat(&mx2);
	cvReleaseMat(&my2);
	*/
#ifdef CvtVideoToImage
	//PreProcessGestureData PreProcessData;
	CString filePath = "G:\\zhipengliu\\dataset\\convertContinousToIsoGestrueTrain";
	CString saveImagePath = "G:\\zhipengliu\\dataset\\DepthVideoToImage";
	CString nframeFile = "G:\\zhipengliu\\dataset\\DepthVideoToImage\\nframe.txt";
	ofstream readFrame;
	readFrame.open(nframeFile, ios::app);
	CString oneVideoSaveImage;
	CString oneImagePath;
	vector<IplImage *> vImage;
	for(int i = 1; i <= 249; i++)
	{
		char num_devel[10];
		sprintf(num_devel, "%03d", i);
		CString oneClassDlevelPath = filePath + "\\" + num_devel;
		//CString outDlevelPath = outputPath + "\\" + num_devel;
		CString oneClassVideoSaveImage = saveImagePath + "\\" + num_devel;

		_mkdir(oneClassVideoSaveImage);//创建图片输出目录
		vector<string> videoPath;
		string subFilePath = oneClassDlevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int videoNum = videoPath.size();
  		for(int j = 0; j < videoNum;)
		{

			string oneDepthVideoPath = videoPath[j++];
			cout<<j<<"  Depth="<<oneDepthVideoPath<<endl;
			string oneRGBVideoPath = videoPath[j++];
			cout<<j<<"  RGB="<<oneRGBVideoPath<<endl;
			CString savename = oneRGBVideoPath.c_str();
			savename = savename.Right(10);
			savename = savename.Left(4);
			oneVideoSaveImage = oneClassVideoSaveImage + "\\" + savename;
			_mkdir(oneVideoSaveImage);
			FCvtVideoToImage(oneDepthVideoPath.c_str(), vImage);
			//写入格式:num_devel(label) savename nframeFile
			int vsize = vImage.size();
			readFrame<<num_devel<<" "<<savename<<" "<<vsize<<endl;
			//release
			
			for(int i = 0; i < vsize; i++)
			{
				char tmp[20];
				sprintf(tmp, "%03d", i);
				oneImagePath = oneVideoSaveImage + "\\" + tmp + ".bmp";
				cvSaveImage(oneImagePath, vImage[i]);
				cvReleaseImage(&vImage[i]);
			}


			for(int i = 0; i < vsize; i++)
				vImage.pop_back();
		}
		
	}
#endif
#ifdef getFrontRGBvideo
	CString frontDepthVideoPath = "E:\\zhipengliu\\dataset\\CvtDepthVideoToFrontVideo";
	CString frontRGBVideoPath = "E:\\zhipengliu\\dataset\\ConGfrontRGBVideo";
	CString RGBVideoPath = "E:\\zhipengliu\\dataset\\convertContinousToIsoGestrueTrain_includehand";
	PreProcessGestureData PreProcessData;
	int nlabel = 249;
	for(int i = 31; i <= nlabel; i++)
	{
		cout<<"label ="<<i<<endl;
		CString CSnlabel;
		CString labelfDepthVideoPath;
		CString labelfRGBvideoPath;
		CString labelRGBvideoPath;

		CSnlabel.Format("%03d", i);

		labelfDepthVideoPath = frontDepthVideoPath + "\\" + CSnlabel;
		labelfRGBvideoPath = frontRGBVideoPath + "\\" + CSnlabel;
		labelRGBvideoPath = RGBVideoPath + "\\" + CSnlabel;

		_mkdir(labelfRGBvideoPath);

		vector<string> vfDepthvideoPath;
		string strtmp = labelfDepthVideoPath.GetBuffer(0);
		getFiles(strtmp, vfDepthvideoPath);

		vector<string> vRGBvideoPath;
		strtmp = labelRGBvideoPath.GetBuffer(0);
		getFiles(strtmp, vRGBvideoPath);
		int nRGBvideo = vRGBvideoPath.size() / 3;
		int RGBbegin = 1;//sample two

		int nfDepthvideo =  vfDepthvideoPath.size();
		int fDbegin = 0;
		for(int j = 0; j < nRGBvideo; j++)
		{
			cout<<j<<endl;
			string oneRGBvideopath = vRGBvideoPath[RGBbegin];
			RGBbegin += 2;
			string onefDepthvideopath = vfDepthvideoPath[fDbegin ++];
			PreProcessData.cvtRGBToFrontRGB(oneRGBvideopath.c_str(), onefDepthvideopath.c_str(), labelfRGBvideoPath);		
			PreProcessData.ReleaseVector();
		}
		int vsize = vfDepthvideoPath.size();
		for(int j = 0; j < vsize; j++)
		{
			vfDepthvideoPath.pop_back();
		}
		vsize = vRGBvideoPath.size();
		for(int j = 0; j < vsize; j++)
		{
			vRGBvideoPath.pop_back();
		}



	}



#endif
#ifdef hard_re
	PreProcessGestureData PreProcessData;
	CString filePath = "E:\\zhipengliu\\dataset\\hard_re";
	CString outputPath = "E:\\zhipengliu\\dataset\\output";
	CString oriFaceDataPath = "F:\\competition\\ProcessingDataConG\\headDataFacePosition";
	for(int i = 1; i <= 15; i++)
	{
		char num_devel[10];
		sprintf(num_devel, "%03d", i);
		CString oneClassDlevelPath = filePath + "\\" + num_devel;
		CString outDlevelPath = outputPath + "\\" + num_devel;
		CString faceDataPath = oriFaceDataPath + "\\" + num_devel;
		_mkdir(outDlevelPath);//创建视频输出目录
		vector<string> videoPath;
		string subFilePath = oneClassDlevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int videoNum = videoPath.size();
  		for(int j = 0; j < videoNum;)
		{

			string oneDepthVideoPath = videoPath[j++];
			cout<<j<<"  Depth="<<oneDepthVideoPath<<endl;
			string oneRGBVideoPath = videoPath[j++];
			cout<<j<<"  RGB="<<oneRGBVideoPath<<endl;
			CString savename = oneRGBVideoPath.c_str();
			savename = savename.Right(10);
			savename = savename.Left(4) + ".txt";
			faceDataPath = faceDataPath + "\\" + savename;
			PreProcessData.faceDataPath = faceDataPath;
			//PreProcessData.GetFileFacePosition();
  			PreProcessData.readVideo(oneRGBVideoPath.c_str(), oneDepthVideoPath.c_str(), outDlevelPath);
  			PreProcessData.ReleaseVector();
			int a = 1;
		}
	}


#endif	
#ifdef get_foreground_info        //Get the foreground area in each video
	PreProcessGestureData PreProcessData;
	string root_path = ValidPath.GetString();
	string output_path = "..\\output\\ForegroundInfo";
	if (_access(output_path.c_str(), 0)==-1)    //表示文件夹不存在
	{
		_mkdir(output_path.c_str());
	}
	vector<string> videoFolders;
	getCurrentDir(root_path, videoFolders);
	for (int j=0; j<videoFolders.size(); j++)
	{
		//创建文件夹 
		if (_access((output_path+"\\"+videoFolders[j]).c_str(), 0)==-1)    //表示文件夹不存在
		{
			_mkdir((output_path+"\\"+videoFolders[j]).c_str());
		}
		string filepath = root_path+"\\"+videoFolders[j]+"\\*M.avi";
		vector<string> videoPath;
		HANDLE hFile;
		LPCTSTR lpFileName = filepath.c_str();
		WIN32_FIND_DATA pNextInfo;	//搜索得到的文件信息将储存在pNextInfo中;
		hFile = FindFirstFile(lpFileName, &pNextInfo);//请注意是 &pNextInfo , 不是pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			//搜索失败
			exit(-1);
		}
		else
		{
			do 
			{
				if(pNextInfo.cFileName[0] == '.')//过滤.和..
					continue;
				videoPath.push_back(filepath.substr(0,filepath.length()-6)+pNextInfo.cFileName);
			} while(FindNextFile(hFile, &pNextInfo));
		}

		int videoNum = videoPath.size();

		for(int k = 0; k < videoNum; k++)
		{
			string oneRGBVideoPath = videoPath[k];
			string oneDepthVideoPath = oneRGBVideoPath;
			int index = oneDepthVideoPath.find_last_of('M');
			oneDepthVideoPath[index] = 'K';
			//cout<<j<<"  Depth="<<oneDepthVideoPath<<endl;		
			//cout<<j<<"  RGB="<<oneRGBVideoPath<<endl;
			int indexOfFile = oneRGBVideoPath.find_last_of('\\');
			CString savename = oneRGBVideoPath.substr(indexOfFile+1, 5).c_str();
			savename = savename+".txt";
			//savename = savename.Right(10);
			//savename = savename.Left(5) + ".txt";
			string foregroundPath = output_path+"\\"+videoFolders[j] + "\\Foreground_" + savename.GetString();
			//PreProcessData.faceDataPath = faceDataPath;
			//PreProcessData.GetFileFacePosition();
			PreProcessData.getForegroundRect(oneRGBVideoPath.c_str(), oneDepthVideoPath.c_str(), foregroundPath);
			PreProcessData.ReleaseVector();
			cout<<videoFolders[j]+"\\Foreground_"+savename.GetString()+" finished"<<endl;
		}
	}

#endif

#ifdef RefineLabel    // refine the OriginalDetectionLabel  1. remove background candidate areas,  2. match candidate areas  3. remove 'large speed'  4. remove 'less side'
	string label_path = "..\\output\\OriginalDetectionLabel";
	string foreground_path = "..\\output\\ForegroundInfo";

	vector<string> video_folders;
	getCurrentDir(label_path, video_folders);
	for (int j=0; j<video_folders.size(); j++)
	{
		//获取每个目录下的检测结果.txt文件
		string txt_path = label_path+"\\"+video_folders[j]+"\\*.txt";
		string foretxt_path = foreground_path+"\\"+video_folders[j];
		vector<string> txts;
		HANDLE hFile;
		LPCTSTR lpFileName = txt_path.c_str();
		WIN32_FIND_DATA pNextInfo;	//搜索得到的文件信息将储存在pNextInfo中;
		hFile = FindFirstFile(lpFileName, &pNextInfo);//请注意是 &pNextInfo , 不是pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			//搜索失败
			continue;
		}
		else
		{
			do 
			{
				if(pNextInfo.cFileName[0] == '.')//过滤.和..
					continue;
				txts.push_back(pNextInfo.cFileName);
			} while(FindNextFile(hFile, &pNextInfo));
		}
		//对每个检测后的结果进行Refine 区分左右手关系
		for (int k=0; k<txts.size(); k++)
		{
			/************************************************************************/
			/* first to remove background candidate area                            */
			/************************************************************************/
			//获取前景区域信息
			//string label_name = txts[k].substr(txts[k].length()-9,5);    for iso
			int indexOfbegin = txts[k].find_last_of('_');
			int indexOfend = txts[k].find('.');
			string label_name = txts[k].substr(indexOfbegin+1, indexOfend-indexOfbegin-1);
			ifstream fore_is;
			vector<Foreground> foregrounds;
			string record = "";
			if(_access((foretxt_path+"\\"+"Foreground_"+label_name+".txt").c_str(), 0)!=-1) //表示文件存在
			{
				fore_is.open(foretxt_path+"\\"+"Foreground_"+label_name+".txt", ios::in);
				while(!fore_is.eof() && fore_is.peek()!=NULL)
				{
					record= "";
					getline(fore_is, record);
					if(record.length()<5)
					{
						continue;
					}
					string id = record.substr(0,4);
					record = record.substr(5, record.length()-5);
					Foreground fore;
					fore.frameID = id;
					istringstream istr(record);
					string pos;
					int count2 = 0;
					while(!istr.eof())
					{
						istr>>pos;
						switch(count2)
						{
						case 0: fore.tlx = atoi(pos.c_str()); count2++; break;
						case 1: fore.tly = atoi(pos.c_str()); count2++; break;
						case 2: fore.brx = atoi(pos.c_str()); count2++; break;
						case 3: fore.bry = atoi(pos.c_str()); count2=0; foregrounds.push_back(fore); break;
						}
					}
				}
			}
			//根据前景区域删除
			ifstream is;
			is.open(txt_path.substr(0, txt_path.length()-5)+txts[k], ios::in);
			//写新文件
			ofstream os;
			os.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine.txt", ios::out);
			if(foregrounds.size()==0)
			{
				record = "";
				while(!is.eof() && is.peek()!=NULL)
				{
					record = "";
					getline(is, record);
					if(record.length()<20)
					{
						continue;
					}
					os<<record<<endl;
				}
				is.close();
				os.close();
			}
			else
			{
				record = "";
				while(!is.eof() && is.peek()!=NULL)
				{
					record = "";
					getline(is, record);
					if(record.length()<20)
					{
						continue;
					}
					string id = record.substr(0,4);			
					record = record.substr(5, record.length()-5);
					//从foregrounds中找到对应的区域
					Foreground fore_frame;
					bool flag = false;
					for (int q=0; q<foregrounds.size(); q++)
					{
						if(foregrounds[q].frameID==id)
						{
							fore_frame = foregrounds[q];
							flag = true;
							break;
						}
					}
					if(!flag)
					{
						os<<id<<" ";
						os<<record<<endl;
						continue;
					}
					CandidateArea rect;
					vector<CandidateArea> now;
					istringstream istr(record);
					string pos;
					int count2 = 0;
					while(!istr.eof())
					{
						istr>>pos;
						switch(count2)
						{
						case 0: rect.tlx = atoi(pos.c_str()); count2++; break;
						case 1: rect.tly = atoi(pos.c_str()); count2++; break;
						case 2: rect.brx = atoi(pos.c_str()); count2++; break;
						case 3: rect.bry = atoi(pos.c_str()); count2=0; now.push_back(rect); break;
						}
					}
					//判断now中每个Candidate Area是否是前景区域
					int q=0;
					while(q<now.size())
					{
						if(!isForeground(fore_frame, now[q]))
						{
							now.erase(now.begin()+q);
						}
						else
						{
							q++;
						}
					}
					if (now.size()!=0)
					{
						os<<id;
						//写入文件
						for (int q=0; q<now.size(); q++)
						{
							CString temp;
							temp.Format(" %03d %03d %03d %03d", now[q].tlx, now[q].tly, now[q].brx, now[q].bry);
							os<<temp;		
						}
						os<<endl;
					}
				}
				is.close();
				os.close();
			}

			/************************************************************************/
			/* second to sure the same hand                                         */
			/************************************************************************/
			//判断根据前景删除后得到的文件是否为空，采用删除后还是之前的
			string filepath = txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine.txt";
			FILE* file;
			fopen_s(&file, filepath.c_str(), "rb");  
			if (file)  
			{  
				int size = _filelength(_fileno(file));  

				if(size==0)
				{
					is.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+".txt", ios::in);
				}
				else
				{
					is.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine.txt", ios::in);
				}
				fclose(file);
			}  
			//写新文件
			os.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine2.txt", ios::out);	
			record="";
			int count = 0;
			vector<CandidateArea> previous;
			while(!is.eof() && is.peek()!=NULL)
			{
				record = "";
				getline(is, record);
				if(record.length()<20)
				{
					continue;
				}
				//对record进行切分
				vector<CandidateArea> now;
				//暂时不考虑前后必须相邻（可以跳跃几帧)
				string id = record.substr(0,4);
				record = record.substr(5, record.length()-5);
				istringstream istr(record);
				string pos;
				CandidateArea rect;
				int count2 = 0;
				while(!istr.eof())
				{
					istr>>pos;
					switch(count2)
					{
					case 0: rect.tlx = atoi(pos.c_str()); count2++; break;
					case 1: rect.tly = atoi(pos.c_str()); count2++; break;
					case 2: rect.brx = atoi(pos.c_str()); count2++; break;
					case 3: rect.bry = atoi(pos.c_str()); count2=0; now.push_back(rect); break;
					}
				}
				if(count==0)
				{
					previous.insert(previous.end(), now.begin(), now.end());
					os<<id;
					for (int p=0; p<min(previous.size(),2); p++)
					{
						CString temp;
						temp.Format(" %03d %03d %03d %03d", previous[p].tlx, previous[p].tly, previous[p].brx, previous[p].bry);
						os<<temp;		
					}
					os<<endl;
					count++;
				}
				else
				{
					//比较now和previous
					os<<id;
					//对previous进行处理 如果其形式为 000 000 000 000 xxx xxx xxx xxx	
					if(previous.size()<=now.size())    //首先
					{
						int preIndex = -1;
						int change = 0;
						int index1 = -1;
						int index2 = -1;
						for (int p=0; p<previous.size(); p++)
						{						
							CandidateArea rect = previous[p];		
							if(rect.tlx==0 && rect.tly==0 && rect.brx==0 && rect.bry==0)
							{
								continue;
							}
							int min = dis(rect, now[change]);
							int index = change;
							for (int q=change+1; q<now.size(); q++)
							{
								if(dis(rect, now[q])<min)
								{
									min = dis(rect, now[q]);
									index = q;
								}
							}
							if(p==0)
							{
								index1 = index;
							}
							if(p==1)
							{
								index2 = index;
							}	
						}
						//进行判断
						if(index1 != -1 && index2 == -1)    //表示previous形式为xxx xxx xxx xxx
						{
							CandidateArea rect = now[index1];
							now.erase(now.begin()+index1);
							now.insert(now.begin(), rect);
						}
						else if(index1 == -1 && index2 != -1)    //表示previous形式为000 000 000 000 xxx xxx xxx xxx
						{
							CandidateArea rect = now[index2];
							now.erase(now.begin()+index2);
							now.insert(now.begin()+1, rect);
						}
						else if(index1 != -1 && index2 !=-1)    //表示previous形式为xxx xxx xxx xxx xxx xxx xxx xxx
						{
							if(index1 != index2)
							{
								CandidateArea rect1 = now[index1];
								CandidateArea rect2 = now[index2];
								now[0] = rect1;
								now[1] = rect2;
							}
							else
							{
								//判断index1/index2离pervious[0]还是previous[1]最近
								CandidateArea rect = now[index1];
								if(dis(previous[0], rect)<=dis(previous[1], rect))
								{
									now.erase(now.begin()+index1);
									now.insert(now.begin(), rect);
									int min = dis(previous[1], now[1]);
									index2 = 1;
									for(int q=2; q<now.size(); q++)
									{
										if(dis(previous[1], now[q])<min)
										{
											min = dis(previous[1], now[q]);
											index2 = q;
										}
									}
									CandidateArea rect2 = now[1];
									now[1] = now[index2];
									now[index2] = rect2;
								}
								else
								{
									now[index1] = now[1];
									now[1] = rect;
									int min = dis(previous[0], now[0]);
									index1 = 0;
									for(int q=2; q<now.size(); q++)
									{
										if(dis(previous[0], now[q])<min)
										{
											min = dis(previous[0], now[q]);
											index1 = q;
										}
									}
									CandidateArea rect2 = now[0];
									now[0] = now[index1];
									now[index1] = rect2;
								}
							}
						}						
						//记录
						for (int p=0; p<min(now.size(), 2); p++)
						{
							CString temp;
							temp.Format(" %03d %03d %03d %03d", now[p].tlx, now[p].tly, now[p].brx, now[p].bry);
							os<<temp;
						}
						//更新previous
						if(previous.size()!=0)
						{
							previous.clear();
						}
						previous.insert(previous.end(), now.begin(), now.begin()+min(now.size(),2));
					}
					else
					{
						if(dis(previous[0], now[0])<=dis(previous[1], now[0]))
						{
							CString temp;
							temp.Format(" %03d %03d %03d %03d", now[0].tlx, now[0].tly, now[0].brx, now[0].bry);
							os<<temp;
							//更新previous
							if(previous.size()!=0)
							{
								previous.clear();
							}
							previous.insert(previous.end(), now.begin(), now.begin()+min(now.size(),2));
						}
						else
						{
							CString temp;
							temp.Format(" %03d %03d %03d %03d %03d %03d %03d %03d", 0, 0, 0, 0, now[0].tlx, now[0].tly, now[0].brx, now[0].bry);
							os<<temp;
							//更新previous
							if(previous.size()!=0)
							{
								previous.clear();
							}
							CandidateArea rect;
							rect.tlx = rect.tly = rect.brx = rect.bry = 0;
							previous.push_back(rect);
							previous.push_back(now[0]);
						}
					}
					os<<endl;
					count++;
				}
			}
			is.close();
			os.close();

			/************************************************************************/
			/* third to remove the large speed frame                                */
			/************************************************************************/
			is.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine2.txt", ios::in);
			os.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine3.txt", ios::out);

			record = "";
			if(previous.size()!=0)
			{
				previous.clear();
			}
			count = 0;
			while(!is.eof() && is.peek()!=NULL)
			{
				record = "";
				getline(is, record);
				if(record.length()<20)
				{
					continue;
				}
				vector<CandidateArea> now;
				//暂时不考虑前后必须相邻（可以跳跃几帧)
				string id = record.substr(0,4);
				record = record.substr(5, record.length()-5);
				istringstream istr(record);
				string pos;
				CandidateArea rect;
				int count2 = 0;
				while(!istr.eof())
				{
					istr>>pos;
					switch(count2)
					{
					case 0: rect.tlx = atoi(pos.c_str()); count2++; break;
					case 1: rect.tly = atoi(pos.c_str()); count2++; break;
					case 2: rect.brx = atoi(pos.c_str()); count2++; break;
					case 3: rect.bry = atoi(pos.c_str()); count2=0;  rect.flag=0; now.push_back(rect); break;
					}
				}
				if(count==0)
				{
					previous.insert(previous.end(), now.begin(), now.end());
					os<<id;
					for (int p=0; p<min(previous.size(),2); p++)
					{
						CString temp;
						temp.Format(" %03d %03d %03d %03d %d", previous[p].tlx, previous[p].tly, previous[p].brx, previous[p].bry, previous[p].flag);
						os<<temp;		
					}
					os<<endl;
					count++;
				}
				else
				{
					os<<id;
					int q = 0;
					while(q<previous.size() && q<now.size())
					{
						if(previous[q].tlx==0 && previous[q].tly==0 && previous[q].brx==0 && previous[q].bry==0)
						{
							previous[q] = now[q];
							q++;
							continue;
						}
						if(now[q].tlx==0 && now[q].tly==0 && now[q].brx==0 && now[q].bry==0)
						{
							previous[q] = now[q];
							q++;
							continue;
						}
						if(centerDis(previous[q], now[q])>=50 || abs(previous[q].tlx-now[q].tlx)>=50 || abs(previous[q].tly-now[q].tly)>=50 || abs(previous[q].brx-now[q].brx)>=50 || abs(previous[q].bry-now[q].bry)>=50)
						{
							//如果对应手跳跃太大, 则删除
							now[q].tlx = 0;
							now[q].tly = 0;
							now[q].brx = 0;
							now[q].bry = 0;
							now[q].flag = 1;
						}
						previous[q] = now[q];
						q++;
					}
					while(q<now.size())
					{
						previous.push_back(now[q]);
						q++;
					}
					for (int p=0; p<now.size(); p++)
					{
						CString temp;
						temp.Format(" %03d %03d %03d %03d %d", now[p].tlx, now[p].tly, now[p].brx, now[p].bry, now[p].flag);
						os<<temp;
					}
					os<<endl;
					count++;
				}
			}
			is.close();
			os.close();

			/************************************************************************/
			/* forth to remove the less detected result                             */
			/************************************************************************/
			is.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine3.txt", ios::in);
			os.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine4.txt", ios::out);

			//统计is文件中左右手数量
			int left = 0;
			int right = 0;
			record = "";
			while (!is.eof() && is.peek()!=NULL)
			{
				record = "";
				getline(is, record);
				if(record.length()<20)
				{
					continue;
				}
				string id = record.substr(0,4);
				record = record.substr(5, record.length()-5);
				istringstream istr(record);
				vector<CandidateArea> now;
				string pos;
				CandidateArea rect;
				int count2 = 0;
				while(!istr.eof())
				{
					istr>>pos;
					switch(count2)
					{
					case 0: rect.tlx = atoi(pos.c_str()); count2++; break;
					case 1: rect.tly = atoi(pos.c_str()); count2++; break;
					case 2: rect.brx = atoi(pos.c_str()); count2++; break;
					case 3: rect.bry = atoi(pos.c_str()); count2++; break;
					case 4: rect.flag = atoi(pos.c_str()); count2=0; now.push_back(rect); break;
					}
				}
				if(now.size()==1)
				{
					left++;
				}
				if(now.size()==2)
				{
					if(!(now[0].tlx==0 && now[0].tly==0 && now[0].brx==0 && now[0].bry==0))
					{
						left++;
					}
					if(!(now[1].tlx==0 && now[1].tly==0 && now[1].brx==0 && now[1].bry==0))
					{
						right++;
					}
				}
			}
			is.close();

			//比较左右侧检测数量比例
			if(left<right*0.3)    //表示左侧远远小于右侧则去掉左侧检测结果
			{
				is.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine3.txt", ios::in);
				record = "";
				while (!is.eof() && is.peek()!=NULL)
				{
					record = "";
					getline(is, record);
					if(record.length()<20)
					{
						continue;
					}
					string id = record.substr(0,4);
					record = record.substr(5, record.length()-5);
					istringstream istr(record);
					vector<CandidateArea> now;
					string pos;
					CandidateArea rect;
					int count2 = 0;
					while(!istr.eof())
					{
						istr>>pos;
						switch(count2)
						{
						case 0: rect.tlx = atoi(pos.c_str()); count2++; break;
						case 1: rect.tly = atoi(pos.c_str()); count2++; break;
						case 2: rect.brx = atoi(pos.c_str()); count2++; break;
						case 3: rect.bry = atoi(pos.c_str()); count2++; break;
						case 4: rect.flag = atoi(pos.c_str()); count2=0; now.push_back(rect); break;
						}
					}
					if (now.size()==2)
					{
						os<<id;
						CString temp;
						temp.Format(" %03d %03d %03d %03d %d %03d %03d %03d %03d %d", 0, 0, 0, 0, 0, now[1].tlx, now[1].tly, now[1].brx, now[1].bry, now[1].flag);
						os<<temp;
						os<<endl;
					}
				}
				is.close();
				os.close();
			}
			else if(right!=0 && right<left*0.3)  //表示右侧远远小于左侧则去掉左侧检测结果
			{
				is.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine3.txt", ios::in);
				record = "";
				while (!is.eof() && is.peek()!=NULL)
				{
					record = "";
					getline(is, record);
					if(record.length()<20)
					{
						continue;
					}
					string id = record.substr(0,4);
					record = record.substr(5, record.length()-5);
					istringstream istr(record);
					vector<CandidateArea> now;
					string pos;
					CandidateArea rect;
					int count2 = 0;
					while(!istr.eof())
					{
						istr>>pos;
						switch(count2)
						{
						case 0: rect.tlx = atoi(pos.c_str()); count2++; break;
						case 1: rect.tly = atoi(pos.c_str()); count2++; break;
						case 2: rect.brx = atoi(pos.c_str()); count2++; break;
						case 3: rect.bry = atoi(pos.c_str()); count2++; break;
						case 4: rect.flag = atoi(pos.c_str()); count2=0; now.push_back(rect); break;
						}
					}
					os<<id;
					CString temp;
					temp.Format(" %03d %03d %03d %03d %d %03d %03d %03d %03d %d", now[0].tlx, now[0].tly, now[0].brx, now[0].bry, now[0].flag, 0, 0, 0, 0, 0);
					os<<temp;
					os<<endl;
				}
				is.close();
				os.close();
			}
			else
			{
				is.open(txt_path.substr(0, txt_path.length()-5)+txts[k].substr(0, txts[k].length()-4)+"_refine3.txt", ios::in);
				record = "";
				while (!is.eof() && is.peek()!=NULL)
				{
					record = "";
					getline(is, record);
					if(record.length()<20)
					{
						continue;
					}
					if(record.length()<=22)
					{
						os<<record;
						CString temp;
						temp.Format(" %03d %03d %03d %03d %d", 0, 0, 0, 0, 0);
						os<<temp<<endl;
					}
					else
					{
						os<<record<<endl;
					}
				}
				is.close();
				os.close();
			}
			cout<<video_folders[j]+"\\"+txts[k]+" finished"<<endl;
		}
	}

#endif

#ifdef getFinalLabel
	string original_path = "..\\output\\OriginalDetectionLabel";
	string final_path = "..\\output\\NeatedDetectionLabel";
	if (_access(final_path.c_str(), 0)==-1)    //表示文件夹不存在
	{
		_mkdir(final_path.c_str());
	}
	vector<string> label_folders;
	getCurrentDir(original_path, label_folders);
	for (int i=0; i<label_folders.size(); i++)
	{
		if (_access((final_path+"\\"+label_folders[i]).c_str(), 0)==-1)    //表示文件夹不存在
		{
			_mkdir((final_path+"\\"+label_folders[i]).c_str());
		}
		string txtPath = original_path+"\\"+label_folders[i]+"\\*refine4.txt";
		vector<string> txts;
		getDefinedFiles(txtPath, txts);
		for (int j=0; j<txts.size(); j++)
		{
			string new_name = txts[j].substr(0,13)+".txt";
			bool flag;
			CopyFile((original_path+"\\"+label_folders[i]+"\\"+txts[j]).c_str(), (final_path+"\\"+label_folders[i]+"\\"+new_name).c_str(), flag);
			cout<<label_folders[i]+"\\"+txts[j]+" finished."<<endl;
		}
	}
#endif

#ifdef ConGSegmentation

	
	CString IsoVideoSaveName = "..\\output\\cvtConGTestToIso";
	CString SegInfopath = "..\\output\\ConGTestSegInfo";
	_mkdir(IsoVideoSaveName);
	_mkdir(SegInfopath);

	vector<string> videoPath;
	int ndevel = getCurrentDir(ValidPath.GetBuffer(0)) ;
	for(int i = 1; i <= ndevel; i++)
	{
		CString CSDevel;
		CSDevel.Format("%03d", i);
		CString ValidDevelPath = ValidPath + "\\" + CSDevel;
		CString IsoVideoDevelPath = IsoVideoSaveName + "\\" + CSDevel;
		CString SegInfoDevelpath = SegInfopath + "\\" + CSDevel;
		CString HandLabelDevel = HandLabelPath + "\\" + CSDevel;
		ifstream fin(IsoVideoDevelPath);
		_mkdir(IsoVideoDevelPath);

		_mkdir(SegInfoDevelpath);
		string subFilePath = ValidDevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int pathlen = videoPath.size();
		int labelnum = getTxtFileNum(videoPath);

		int videoNum = (videoPath.size() - labelnum)/ 2;

		int labelbegin = videoNum * 2;
		int numrgbvideo = videoNum;
		int rgbbegin = 1 + 0 * 2;
		for(int j = 0; j < numrgbvideo; j++)
		{
			if(j == 7)
			{
				int tmp = 0;
			}
			vector<IsoVideoInfo> cvtIsovideoInfo;
			CString rgbFilePath = videoPath[rgbbegin].c_str();
			rgbbegin += 2;
			CString handLabelFilePath = HandLabelDevel + "\\Label_" + rgbFilePath.Right(11).Left(7) + ".txt";
			if(!IsCorrespondingFile(rgbFilePath, handLabelFilePath))
			{
				CString tmp = rgbFilePath;
				//of<<tmp.Right(14)<<endl;
				continue;
			}
			//double a = CvtConvideoToIsovideo(handLabelFilePath, rgbFilePath, IsoVideoDevelPath, cvtIsovideoInfo, SegInfoDevelpath);
			double a = CvtConvideoToIsovideo_flag(handLabelFilePath, rgbFilePath, IsoVideoDevelPath, cvtIsovideoInfo, SegInfoDevelpath);
			int tmpsize = cvtIsovideoInfo.size();
			for(int k = 0; k < tmpsize; k++)
			{
				cvtIsovideoInfo.pop_back();
			}
		}
		int vsize = videoPath.size();
		for(int j = 0; j < vsize; j ++)
		{
			videoPath.pop_back();
		}
	}
#endif

#ifdef getTestFacePosition
	//写入格式：脸部中心坐标：(frame, x0, y0):x0 y0 每行一帧图片,若无法检测到人脸，则x0=y0=0;
	//			int temp=(int)((uchar*)(tempImg->imageData+y0*tempImg->widthStep))[x0];
	//			minDepth += temp;
	//

	PreProcessGestureData PreProcessFace;
	CString filePath = ValidPath;
	CString outputPath = "..\\output\\RGBFacePosition";
	_mkdir(outputPath);
	fstream filestream;
	fstream filestream_con5;
	int num_file = ndevel;//249;
	int count = 0;
	for(int i = 1; i <= num_file; i++)
	{
		
		char num_devel[10];
		sprintf(num_devel, "%03d", i);
		CString oneClassDlevelPath = filePath + "\\" + num_devel;
		CString outDlevelPath = outputPath + "\\" + num_devel;


		_mkdir(outDlevelPath);//创建head中心点输出目录
		vector<string> videoPath;
		string subFilePath = oneClassDlevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int videoNum = videoPath.size();
		//vector<IplImage*> vColor;//vColor 存储RGB video
		//cvNamedWindow("ShowImage", CV_WINDOW_AUTOSIZE);
		
		for(int j = 0; j < videoNum;)
		{
			string oneDepthVideoPath = videoPath[j++];//越过DepthVideo
			string oneRGBVideoPath = videoPath[j++];
					
			CString savename = oneRGBVideoPath.c_str();
			savename = savename.Right(10);
			savename = savename.Left(4) + ".txt";
			CString headfilePath = outDlevelPath + "\\" + savename;
			filestream.open(headfilePath, ios::in);
			if(filestream)
			{
				filestream.close();
				continue;
			}
			cout<<"RGB="<<oneRGBVideoPath<<endl;
			CString finalFacePath;
			CvCapture *capture = cvCreateFileCapture(oneRGBVideoPath.c_str());
			int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
			if(numFrames < 1)
			{
				cout<<"video file's path is wrong and can be read"<<endl;
				//PreProcessFace.ReleaseVector();
				continue;
			}		
			finalFacePath = headfilePath;
			for(int i = 0; i < numFrames; i++)
			{
				IplImage* tempFrame = cvQueryFrame(capture);
				//cvShowImage("ShowImage", tempFrame);
				//cvWaitKey(100);
				//cout<<i + 1<<endl;
				int countframe = i + 1;
				
				if(PreProcessFace.OutheadDetectionVIPLSDK(tempFrame, finalFacePath, countframe))
					break;
				if(i == numFrames - 1)
				{
					PreProcessFace.readVideo(oneRGBVideoPath.c_str(), oneDepthVideoPath.c_str());
					PreProcessFace.getFacePositionHist(finalFacePath);
					PreProcessFace.ReleaseVector();
				}

					
			}
			filestream.close();
			PreProcessFace.ReleaseVector();
			cvReleaseCapture(&capture);
		}
	}	
#endif
#ifdef GetIsoFacePosition
	//写入格式：脸部中心坐标：(frame, x0, y0):x0 y0 每行一帧图片,若无法检测到人脸，则x0=y0=0;
	//			int temp=(int)((uchar*)(tempImg->imageData+y0*tempImg->widthStep))[x0];
	//			minDepth += temp;
	//
	PreProcessGestureData PreProcessFace;
	
	CString outputPath = "..\\output\\IsoRGBtestFacePosition.txt";
	
	int videoLabel = -1;

	fstream trainfilestream;
	trainfilestream.open(Trainlist, ios::in);
	char tbuffer[256];
	vector<string> vdepthtrainlist;
	vector<int> vlabel;
	char rgbname[100], depthname[100];
	int label;
	char cdevel[100], cvideoID[100];
	char ctmp[100];
	while(trainfilestream.getline(tbuffer, 256))
	{
		//sscanf(tbuffer, "%s %s %d", rgbname, depthname, &label);
		sscanf(tbuffer, "%s %s", rgbname, depthname);
		label = 0;
		string sdevel;
		sdevel.assign(rgbname,5,3); 
		string svideoID;
		svideoID.assign(rgbname, 9, 11);
		
		string stmp = sdevel + "\\"  + svideoID;
		std::cout<<stmp<<endl;
		vdepthtrainlist.push_back(stmp);
		vlabel.push_back(label);

		string dsdevel;
		dsdevel.assign(depthname,5,3); 
		string dsvideoID;
		dsvideoID.assign(depthname, 9, 11);
		
		string dstmp = dsdevel + "\\"  + dsvideoID;
		std::cout<<dstmp<<endl;
		vdepthtrainlist.push_back(dstmp);
		vlabel.push_back(label);
	}
	
	int numVideo = vdepthtrainlist.size();
	for(int i = 0; i < numVideo; i++)
	{
		std::cout<<i<<"/"<<numVideo<<endl;
		string oneRGBVideoPath = filePath + "\\" + vdepthtrainlist[i];
		cout<<"RGB:"<<vdepthtrainlist[i]<<endl;
		i++;
		string oneDepthVideoPath = filePath + "\\" + vdepthtrainlist[i];
		cout<<"Depth:"<<vdepthtrainlist[i]<<endl;
		CvCapture *capture = cvCreateFileCapture(oneRGBVideoPath.c_str());
		int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
		if(numFrames < 1)
		{
			cout<<"video file's path is wrong and can be read"<<endl;
			//PreProcessFace.ReleaseVector();
			continue;
		}		
		CString finalFacePath;
		finalFacePath = outputPath;
		CString tmp = vdepthtrainlist[i].c_str();
		CString CSvideoid = tmp.Right(9).Left(5);
		int videoid = _ttoi(CSvideoid);
		cout<<"video id:"<<videoid<<endl;
		for(int i = 0; i < numFrames; i++)
		{
			IplImage* tempFrame = cvQueryFrame(capture);
			int countframe = i + 1;
			if(PreProcessFace.IsoOutheadDetectionVIPLSDK(tempFrame, finalFacePath, countframe, videoid))
				break;
			if(i == numFrames - 1)
			{
				PreProcessFace.readVideo(oneRGBVideoPath.c_str(), oneDepthVideoPath.c_str());
				PreProcessFace.getISoFacePositionHist(finalFacePath, countframe, videoid);
				PreProcessFace.ReleaseVector();
			}

					
		}
		PreProcessFace.ReleaseVector();
		cvReleaseCapture(&capture);
	}
	for(int i = numVideo - 1; i >=0 ; i--)
	{
		vlabel.pop_back();
		vdepthtrainlist.pop_back();
	}
	vdepthtrainlist.clear();
	vlabel.clear();
#endif
#ifdef testOneVideo
	CString colorVedioPath("F:\\competition\\ProcessingDataConG\\convertContinousToIsoGestrueTrain\\005\\0001.M.avi");
	CString headfilePath("F:\\competition\\ProcessingDataConG\\testoutput\\face_con=5_5_1.txt");
	CString depthVedioPath("F:\\competition\\ProcessingDataConG\\convertContinousToIsoGestrueTrain\\test\\0012.K.avi");
	PreProcessGestureData PreProcessData;
	CvCapture *capture = cvCreateFileCapture(colorVedioPath);
	cvNamedWindow("ShowImage", CV_WINDOW_AUTOSIZE);
	int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
	if(numFrames < 1)
	{
		cout<<"video file's path is wrong and can be read"<<endl;
			return 0;
	}		
	for(int i = 0; i < numFrames; i++)
	{
		IplImage* tempFrame = cvQueryFrame(capture);
		cvShowImage("ShowImage", tempFrame);
		cvWaitKey(100);
		cout<<i + 1<<endl;
		if(PreProcessData.OutheadDetectionVIPLSDK(tempFrame, headfilePath, i + 1))
			break;
	}
	//CString RGBVideo = "F:\\competition\\ProcessingDataConG\\hard_re\\001\\0002.M.avi";
	//CString DepthVideo = "F:\\competition\\ProcessingDataConG\\hard_re\\001\\0002.K.avi";
	//CString outDlevelPath = "F:\\competition\\ProcessingDataConG\\convertContinousToIsoGestrueTrain";
	//vector<string> videoPath;
	//string subFilePath = outDlevelPath.GetBuffer(0);
	//getFiles(subFilePath, videoPath);
	//PreProcessData.readVideo(RGBVideo, DepthVideo, outDlevelPath);	
#endif

#ifdef visualizationHandDetection
	CString ValidPath = "E:\\Visualization\\Valid\\oridata\\2";
	CString IsoVideoSaveName = "E:\\Visualization\\Valid\\depthvisual\\2";
	vector<string> videoPath;
	int ndevel = 10;
	for(int i = 1; i <= 1; i++)
	{
		CString CSDevel;
		CSDevel.Format("%03d", i);
		CString ValidDevelPath = ValidPath + "\\" + CSDevel;
		CString IsoVideoDevelPath = IsoVideoSaveName + "\\" + CSDevel;
		_mkdir(IsoVideoDevelPath);

		string subFilePath = ValidDevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int pathlen = videoPath.size();
		int numrgbvideo = pathlen / 3;
		int Labelbegin = numrgbvideo + 0;
		int rgbbegin = numrgbvideo * 2;
		int depthbegin = 0;
		for(int j = 0; j < numrgbvideo; j++)
		{
			CString rgbFilePath = videoPath[rgbbegin++ ].c_str();
			CString depthfilepath = videoPath[depthbegin++].c_str();
			CString handLabelFilePath = videoPath[Labelbegin++ ].c_str();
			double a = fvisualHandDetection(handLabelFilePath, depthfilepath, IsoVideoDevelPath);

		}
		int vsize = videoPath.size();
		for(int j = 0; j < vsize; j ++)
		{
			videoPath.pop_back();
		}
	}
#endif
#ifdef GetContinuousGestureHog
	CString filePath = "E:\\zhipengliu\\dataset\\Continuous Gesture\\convertContinousToIsoGestrueTrain_new";
	CString saveImagePath = "E:\\zhipengliu\\dataset\\Continuous Gesture\\version3\\NeatedRGBHog\\train";
	CString LabelHandFile = "E:\\zhipengliu\\dataset\\Continuous Gesture\\version3\\handlabel\\Cons\\convertContinousToIsoGestrueTrain";
	int videoLabel = -1;
	int ndevel = 249;
	CString losthandDetection = "E:\\ConNeatedtrainlosthand.txt";
	ofstream of;
	of.open(losthandDetection, ios::app);

	for(int i = 1; i <= ndevel; i++)
	{
		videoLabel = i;// train
		//videoLabel = 0;//test
		char num_devel[10];
		sprintf(num_devel, "%03d", i);
		CString oneClassDlevelPath = filePath + "\\" + num_devel;
		//CString outDlevelPath = outputPath + "\\" + num_devel;
		CString oneClassLabelpath = LabelHandFile + "\\" + num_devel;

		vector<string> videoPath;
		string subFilePath = oneClassDlevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int labelnum = getTxtFileNum(videoPath);

		int videoNum = (videoPath.size() - labelnum)/ 2;
		int rgbvideobegin = 1;
		int Depthvideobegin = 0;
		int labelbegin = videoNum * 2;
		std::cout<<"labelbegin:"<<labelbegin<<endl;
  		for(int j = 1; j < videoNum + 1;j++)
		{
		
			string oneRGBVideoPath = videoPath[rgbvideobegin];
			rgbvideobegin = rgbvideobegin + 2;
			string oneDepthVideoPath = videoPath[Depthvideobegin];
			Depthvideobegin += 2;
			CString ctmp = oneRGBVideoPath.c_str();//
			//string LabelHandPath = videoPath[labelbegin];
			CString LabelHandFinallPath = oneClassLabelpath + "\\Label_" + ctmp.Right(10).Left(6) + ".txt";
			//CString LabelHandFinallPath = oneClassLabelpath + "\\Label_" + ctmp.Right(11).Left(7) + ".txt";//valid

			
			if(!IsCorrespondingFile(oneRGBVideoPath.c_str(), LabelHandFinallPath))
			{
				CString tmp = oneRGBVideoPath.c_str();
				std::cout<<endl<<endl;
				of<<tmp.Right(14)<<endl;
				continue;
			}
			
			labelbegin ++;
			std::cout<<"RGB:"<<oneRGBVideoPath<<endl;
			std::cout<<"HandLabel:"<<LabelHandFinallPath<<endl;				
			char savename[50];
			sprintf(savename, "HOG_%03d_%04d.txt", i, j);
			//CString savefinalpath = saveImagePath + "\\" + "HOG_" + ctmp.Right(11).Left(5) + ".txt";
			CString savefinalpath = saveImagePath + "\\" + savename;
			//ExtractHogFeature(oneRGBVideoPath.c_str(), LabelHandFinallPath, videoLabel, savefinalpath);
			ExtractHogFeature_flag(oneRGBVideoPath.c_str(), LabelHandFinallPath, videoLabel, savefinalpath);

		}
		int vsize = videoPath.size();
		for(int ti = 0; ti < vsize; ti++)
			videoPath.pop_back();
	}
	of.close();
#endif

#ifdef cvtRGBface2Depthface
	fgenerateMap();//create memory
	filePath = "..\\output\\RGBFacePosition";
	CString savePath = "..\\output\\DepthFacePosition";
	_mkdir(savePath);
	int videoLabel = -1;
	//int ndevel = 249;
	ofstream writestream;
	fstream trainfilestream;
	for(int i = 1; i <= ndevel; i++)
	{
		cout<<i<<endl;
		videoLabel = i;// train
		//videoLabel = 0;
		char num_devel[10];
		sprintf(num_devel, "%03d", i);
		CString oneClassDlevelPath = filePath + "\\" + num_devel;
		//CString outDlevelPath = outputPath + "\\" + num_devel;
		CString oneClasssavePath = savePath + "\\" + num_devel;
		_mkdir(oneClasssavePath);
		vector<string> videoPath;
		string subFilePath = oneClassDlevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int videoNum = videoPath.size();
  		for(int j = 0; j < videoNum;j++)
		{
			string oneRGBlabelpath = videoPath[j];
			CString CStmp = oneRGBlabelpath.c_str();
			CString oneDepthLabel = oneClasssavePath + "\\" + CStmp.Right(8);
			writestream.open(oneDepthLabel, ios::out|ios::trunc);//write
			trainfilestream.open(oneRGBlabelpath, ios::in);
			char tbuffer[256];
			int id, frame, x0, y0;
			TwoDimension A;
			trainfilestream.getline(tbuffer, 256);
			sscanf(tbuffer, "%d %d",&x0, &y0);
			A.x = x0;
			A.y = y0;
			RGB2DepthRectify(A, A);
			char ctmp[100];
			sprintf(ctmp, "%d %d\n",A.x, A.y);
			writestream<<ctmp;
			trainfilestream.close();
			writestream.close();
		}
		int vsize = videoPath.size();
		for(int ti = 0; ti < vsize; ti++)
			videoPath.pop_back();
	}
	cvReleaseMat(&mx1);
	cvReleaseMat(&my1);
	cvReleaseMat(&mx2);
	cvReleaseMat(&my2);
#endif
#ifdef GetContinuousTestGesture
	filePath = ValidPath;//"E:\\zhipengliu\\dataset\\Continuous Gesture\\ConGD_phase_2\\test";
	CString saveImagePathM = "..\\output\\HOG\\RGB";
	CString saveImagePathK = "..\\output\\HOG\\Depth";
	_mkdir("..\\output\\HOG");
	_mkdir(saveImagePathM);
	_mkdir(saveImagePathK);
	CString LabelHandFile = HandLabelPath;//"E:\\zhipengliu\\OriginalDetectionLabel\\Continues\\test";
	videoLabel = -1;
	//int ndevel = 81;
	CString losthandDetection = "..\\output\\ConTestlosthand.txt";
	ofstream of;
	of.open(losthandDetection, ios::app);

	for(int i = 1; i <= ndevel; i++)
	{
		videoLabel = 0;
		char num_devel[10];
		sprintf(num_devel, "%03d", i);
		CString oneClassDlevelPath = filePath + "\\" + num_devel;
		//CString outDlevelPath = outputPath + "\\" + num_devel;
		CString oneClassLabelpath = LabelHandFile + "\\" + num_devel;

		vector<string> videoPath;
		string subFilePath = oneClassDlevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);
		int labelnum = getTxtFileNum(videoPath);

		int videoNum = (videoPath.size() - labelnum)/ 2;
		int rgbvideobegin = 1;
		int Depthvideobegin = 0;
		int labelbegin = videoNum * 2;
		cout<<"labelbegin:"<<labelbegin<<endl;
  		for(int j = 1; j < videoNum + 1;j++)
		{
		
			string oneRGBVideoPath = videoPath[rgbvideobegin];
			rgbvideobegin = rgbvideobegin + 2;
			string oneDepthVideoPath = videoPath[Depthvideobegin];
			Depthvideobegin += 2;
			CString ctmp = oneDepthVideoPath.c_str();
			//string LabelHandPath = videoPath[labelbegin];
			//CString LabelHandFinallPath = oneClassLabelpath + "\\Label_" + ctmp.Right(10).Left(6) + ".txt";
			CString LabelHandFinallPath = oneClassLabelpath + "\\Label_" + ctmp.Right(11).Left(7) + ".txt";//valid

			
			if(!IsCorrespondingFile(oneRGBVideoPath.c_str(), LabelHandFinallPath))
			{
				CString tmp = oneRGBVideoPath.c_str();
				cout<<endl<<endl;
				of<<tmp.Right(14)<<endl;
				continue;
			}
			
			labelbegin ++;
			cout<<oneDepthVideoPath<<endl;
			cout<<"HandLabel:"<<LabelHandFinallPath<<endl;				
			char savename[50];
			sprintf(savename, "HOG_%03d_%04d.txt", i, j);
			//CString savefinalpath = saveImagePath + "\\" + "HOG_" + ctmp.Right(11).Left(5) + ".txt";
			CString savefinalpath = saveImagePathK + "\\" + savename;
			ExtractHogFeature(oneDepthVideoPath.c_str(), LabelHandFinallPath, videoLabel, savefinalpath);

			ctmp = oneRGBVideoPath.c_str();
			savefinalpath = saveImagePathM + "\\" + savename;
			LabelHandFinallPath = oneClassLabelpath + "\\Label_" + ctmp.Right(11).Left(7) + ".txt";
			ExtractHogFeature(oneRGBVideoPath.c_str(), LabelHandFinallPath, videoLabel, savefinalpath);

		}
		int vsize = videoPath.size();
		for(int ti = 0; ti < vsize; ti++)
			videoPath.pop_back();
	}
	of.close();
#endif

#ifdef GetIsoGestureHog
	//string filePath = "E:\\zhipengliu\\dataset\\IsoGesture\\IsoGD_Phase_1\\IsoGD_phase_1\\train";
	CString saveImagePath = "..\\output\\depthHOG";
	
	//CString Trainlist = "E:\\research\\competetion\\IsoGesture\\IsoGD_Phase_1\\IsoGD_phase_1\\train_list.txt";
	videoLabel = -1;

	trainfilestream;
	trainfilestream.open(Trainlist, ios::in);
	tbuffer[256];

	while(trainfilestream.getline(tbuffer, 256))
	{
		sscanf(tbuffer, "%s %s %d", rgbname, depthname);
		//sscanf(tbuffer, "%s %s", rgbname, depthname);
		label = 0;
		string sdevel;
		sdevel.assign(rgbname,6,3); 
		string svideoID;
		svideoID.assign(rgbname, 10, 11);
		
		string stmp = sdevel + "\\"  + svideoID;
		std::cout<<stmp<<endl;
		vdepthtrainlist.push_back(stmp);
		vlabel.push_back(label);
	}
	numVideo = vdepthtrainlist.size();
	for(int i = 0; i < numVideo; i++)
	{
		//if (i == 14215)//train
		//	continue;
		std::cout<<i<<"/"<<numVideo<<endl;
		string oneDepthVideoPath = filePath + "\\" + vdepthtrainlist[i];
		label = vlabel[i];
		CString CStmp = oneDepthVideoPath.c_str();
		if(i == 17)
			int tmp = 1;
		//CString oneDepthLabelPath = LabelHandFile + "\\" + CStmp.Right(15).Left(3) + "\\Label_" + CStmp.Right(9).Left(5) + ".txt";//RGB and Depth
		//CString oneDepthLabelPath = LabelHandFile + "\\"  + "Label_" + CStmp.Right(9).Left(5) + ".txt";
		CString FinalSavePath = saveImagePath + "\\" + "HOG_" + CStmp.Right(9).Left(5) + ".txt";
		CString oneDepthLabelPath = LabelHandFile + "\\" + CStmp.Right(15).Left(3) + "\\Label_K_" + CStmp.Right(9).Left(5) + ".txt";
		ExtractHogFeature(oneDepthVideoPath.c_str(), oneDepthLabelPath, label, FinalSavePath);
		//ExtractHogFeature_flag(oneDepthVideoPath.c_str(), oneDepthLabelPath, label, FinalSavePath);

	}
	trainfilestream.close();
	for(int i = numVideo - 1; i >=0 ; i--)
	{
		vlabel.pop_back();
		vdepthtrainlist.pop_back();
	}
	vdepthtrainlist.clear();
	vlabel.clear();

	//string filePath = "E:\\zhipengliu\\dataset\\IsoGesture\\IsoGD_Phase_1\\IsoGD_phase_1\\train";
	saveImagePath = "..\\output\\RGBHOG";
	
	//CString Trainlist = "E:\\research\\competetion\\IsoGesture\\IsoGD_Phase_1\\IsoGD_phase_1\\train_list.txt";
	videoLabel = -1;

	trainfilestream.open(Trainlist, ios::in);

	while(trainfilestream.getline(tbuffer, 256))
	{
		sscanf(tbuffer, "%s %s %d", rgbname, depthname);
		//sscanf(tbuffer, "%s %s", rgbname, depthname);
		label = 0;
		string sdevel;
		sdevel.assign(depthname,6,3); 
		string svideoID;
		svideoID.assign(depthname, 10, 11);
		
		string stmp = sdevel + "\\"  + svideoID;
		std::cout<<stmp<<endl;
		vdepthtrainlist.push_back(stmp);
		vlabel.push_back(label);
	}
	numVideo = vdepthtrainlist.size();
	for(int i = 0; i < numVideo; i++)
	{
		//if (i == 14215)//train
		//	continue;
		std::cout<<i<<"/"<<numVideo<<endl;
		string oneDepthVideoPath = filePath + "\\" + vdepthtrainlist[i];
		label = vlabel[i];
		CString CStmp = oneDepthVideoPath.c_str();
		if(i == 17)
			int tmp = 1;
		//CString oneDepthLabelPath = LabelHandFile + "\\" + CStmp.Right(15).Left(3) + "\\Label_" + CStmp.Right(9).Left(5) + ".txt";//RGB and Depth
		//CString oneDepthLabelPath = LabelHandFile + "\\"  + "Label_" + CStmp.Right(9).Left(5) + ".txt";
		CString FinalSavePath = saveImagePath + "\\" + "HOG_" + CStmp.Right(9).Left(5) + ".txt";
		CString oneDepthLabelPath = LabelHandFile + "\\" + CStmp.Right(15).Left(3) + "\\Label_M_" + CStmp.Right(9).Left(5) + ".txt";
		ExtractHogFeature(oneDepthVideoPath.c_str(), oneDepthLabelPath, label, FinalSavePath);
		//ExtractHogFeature_flag(oneDepthVideoPath.c_str(), oneDepthLabelPath, label, FinalSavePath);

	}
	trainfilestream.close();
	for(int i = numVideo - 1; i >=0 ; i--)
	{
		vlabel.pop_back();
		vdepthtrainlist.pop_back();
	}
	vdepthtrainlist.clear();
	vlabel.clear();

#endif
#ifdef CvtRGBhandLabelToDepthLabel
	fgenerateMap();//create memory
	CString ValidPath = "F:\\research\\competetion\\IsoGesture\\version2\\IsoTrainValidHandDetectVersion2\\valid";
	CString IsoVideoSaveName = "F:\\research\\competetion\\IsoGesture\\version2\\RGB2DepthHandDetectionMap\\valid";
	vector<string> videoPath;
	int ndevel = 29;
	for(int i = 1; i <= ndevel; i++)
	{
		std::cout<<"ndevel:"<<i<<endl;
		CString CSDevel;
		CSDevel.Format("%03d", i);
		CString ValidDevelPath = ValidPath + "\\" + CSDevel;
		CString IsoVideoDevelPath = IsoVideoSaveName;
		//_mkdir(IsoVideoDevelPath);

		string subFilePath = ValidDevelPath.GetBuffer(0);
		getFiles(subFilePath, videoPath);

		int labelnum = getTxtFileNum(videoPath);

		int videoNum = (videoPath.size() - labelnum)/ 2;

		int Labelbegin = videoNum;
		int rgbbegin = videoNum + labelnum;
		int depthbegin = 0;
		for(int j = 0; j < labelnum; j++)
		{
			//CString rgbFilePath = videoPath[rgbbegin++ ].c_str();
			//CString depthfilepath = videoPath[depthbegin++].c_str();
			CString handLabelFilePath = videoPath[Labelbegin++ ].c_str();
			CString LabelSavePath = IsoVideoDevelPath + "\\" + handLabelFilePath.Right(15);
			cvtRGBHandLabel2Depth(handLabelFilePath, LabelSavePath);
			//double a = fvisualHandDetection(handLabelFilePath, depthfilepath, IsoVideoDevelPath);

		}
		int vsize = videoPath.size();
		for(int j = 0; j < vsize; j ++)
		{
			videoPath.pop_back();
		}
	}
	cvReleaseMat(&mx1);
	cvReleaseMat(&my1);
	cvReleaseMat(&mx2);
	cvReleaseMat(&my2);
#endif
	return 0;
}
double gethandPosition(frameHandPosition hand, int frame)
{
	if(hand.isDetectOneHand)
		return (hand.handA2.y + hand.handA1.y) / 2.0;
	else                            //返回两只手中位置较高的位置
	{
		double handAcenter = (hand.handA1.y + hand.handA2.y) / 2.0;
		double handBcenter = (hand.handB1.y + hand.handB2.y) / 2.0;
		if(frame != 0)				//actioning
			if(handAcenter < handBcenter)
				return handAcenter;
			else
				return handBcenter;
		else
		{
		//直接取两个手中心的平均高度,防止开始手部区域故意抬高
			return (handAcenter + handBcenter) / 2;
		}

	}

}
double fvisualHandDetection(CString handLabelFilePath, CString rgbFilePath, CString IsoVideoSaveName)
{
	CString videoname = rgbFilePath.Right(11);
	IsoVideoSaveName = IsoVideoSaveName + "\\" + "VH" + videoname;

	std::cout<<videoname<<endl;
	fstream filestream;

	map <int, frameHandPosition> maphand;
	
	frameHandPosition handposition;
	vector<IplImage*> vColor;//vColor 存储RGB video
	char buffer[256];

	filestream.open(handLabelFilePath, ios::in);
	if(!filestream)
	{
		std::cout<<handLabelFilePath<<" does not exists"<<endl;
		return -1;
	}
	//read handLabelFilePath
	while(filestream.getline(buffer, 256))
	{
		int bufferlen = strlen(buffer);
		if(bufferlen <= 20)//detect one hand
		{
			handposition.isDetectOneHand = true;
			sscanf(buffer, "%04d %03d %03d %03d %03d\n", &handposition.nframe, &handposition.handA1.x, &handposition.handA1.y, &handposition.handA2.x, &handposition.handA2.y);
			TwoDimension tmp;
			RGB2DepthRectify(handposition.handA1, handposition.handA1);
			RGB2DepthRectify(handposition.handA2, handposition.handA2);
			handposition.handB1.x = 0;
			handposition.handB1.y = 0;
			handposition.handB2.x = 0;
			handposition.handB2.y = 0;
		}
		else//detect two hands
		{
			handposition.isDetectOneHand = false;
			sscanf(buffer, "%04d %03d %03d %03d %03d %03d %03d %03d %03d\n", &handposition.nframe, &handposition.handA1.x, &handposition.handA1.y, &handposition.handA2.x, &handposition.handA2.y,
				&handposition.handB1.x, &handposition.handB1.y, &handposition.handB2.x, &handposition.handB2.y);
			RGB2DepthRectify(handposition.handA1, handposition.handA1);
			RGB2DepthRectify(handposition.handA2, handposition.handA2);
			RGB2DepthRectify(handposition.handB1, handposition.handB1);
			RGB2DepthRectify(handposition.handB2, handposition.handB2);
		}
		maphand[handposition.nframe] = handposition;

	}
	// read video
	CvCapture *capture = cvCreateFileCapture(rgbFilePath);
  	int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
	if(numFrames < 1)
		return -1;
	for(int i = 0; i < numFrames; i++)
	{
		IplImage* tempFrame = cvQueryFrame(capture);
		vColor.push_back(cvCloneImage(tempFrame));
	}
	cvReleaseCapture(&capture);

	CvVideoWriter *writerhand = 0;
	writerhand = cvCreateVideoWriter(IsoVideoSaveName, CV_FOURCC('M','P','4','2'), 10, cvSize(vColor[0]->width, vColor[0]->height), 1);
	for(int i = 0; i < numFrames; i++)
	{
		
		if (maphand.find(i) != maphand.end())
		{
			handposition = maphand.find(i)->second;
			TwoDimension A, B;
			A = handposition.handA1;
			B = handposition.handA2;
			cvRectangle(vColor[i], cvPoint(A.x,A.y), cvPoint(B.x, B.y), Scalar(0, 255, 255), 1, 1, 0);  
			if(handposition.isDetectOneHand == false)
			{
				A = handposition.handB1;
				B = handposition.handB2;
				cvRectangle(vColor[i], cvPoint(A.x,A.y), cvPoint(B.x, B.y), Scalar(0,255,0), 1, 1, 0);  
			}
		}
		cvWriteFrame(writerhand,vColor[i]);
	}
	cvReleaseVideoWriter(&writerhand);
	//delete maphand
	map<int, frameHandPosition>::iterator it;
	for(it = maphand.begin(); it != maphand.end();)
	{
		maphand.erase(it++);
	}
	//release video vector memory
	for(int i = numFrames - 1; i >= 0; i--)
	{
		cvReleaseImage(&vColor[i]);
		vColor.pop_back();
	}
	return 0;
}

double CvtConvideoToIsovideo(CString handLabelFilePath, CString rgbFilePath, CString IsoVideoSaveName, vector<IsoVideoInfo>& cvtIsovideoInfo, CString SegInfoDevelpath)
{
	CString videoname = rgbFilePath.Right(11).Left(5);
	IsoVideoSaveName = IsoVideoSaveName + "\\" + videoname;

	_mkdir(IsoVideoSaveName);

	CString IsoinfoTxt = SegInfoDevelpath + "\\" + videoname + ".txt";
	ofstream of;
	of.open(IsoinfoTxt, ios::app);

	std::cout<<videoname<<endl;
	fstream filestream;

	map <int, frameHandPosition> maphand;
	
	frameHandPosition handposition;
	vector<IplImage*> vColor;//vColor 存储RGB video
	char buffer[256];

	filestream.open(handLabelFilePath, ios::in);
	if(!filestream)
	{
		std::cout<<handLabelFilePath<<" does not exists"<<endl;
		return -1;
	}
	//read handLabelFilePath
	while(filestream.getline(buffer, 256))
	{
		int bufferlen = strlen(buffer);
		if(bufferlen <= 20)//detect one hand
		{
			handposition.isDetectOneHand = true;
			sscanf(buffer, "%04d %03d %03d %03d %03d\n", &handposition.nframe, &handposition.handA1.x, &handposition.handA1.y, &handposition.handA2.x, &handposition.handA2.y);
			handposition.handB1.x = 0;
			handposition.handB1.y = 0;
			handposition.handB2.x = 0;
			handposition.handB2.y = 0;
		}
		else//detect two hands
		{
			handposition.isDetectOneHand = false;
			sscanf(buffer, "%04d %03d %03d %03d %03d %03d %03d %03d %03d\n", &handposition.nframe, &handposition.handA1.x, &handposition.handA1.y, &handposition.handA2.x, &handposition.handA2.y,
				&handposition.handB1.x, &handposition.handB1.y, &handposition.handB2.x, &handposition.handB2.y);
		}
		maphand[handposition.nframe] = handposition;

	}
	// read video
	CvCapture *capture = cvCreateFileCapture(rgbFilePath);
  	int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
	if(numFrames < 1)
		return -1;
	for(int i = 0; i < numFrames; i++)
	{
		IplImage* tempFrame = cvQueryFrame(capture);
		vColor.push_back(cvCloneImage(tempFrame));
	}
	cvReleaseCapture(&capture);

	int gestureStart = 0;//the frame indicates the one gesture begins
	int gestureEnd = 0;//the frame indicates the one gesture ends
	double level = 0;//the level can judge if the gesture begins 
	//IplImage* tempImg = cvCreateImage(cvSize(vColor[0]->width,vColor[0]->height),vColor[0]->depth,vColor[0]->nChannels);
	double threshold = 10;
	int videocount = 0;
	CvVideoWriter *writer = 0;

	CvVideoWriter *writerhand = 0;
	CString CStmp;
	//CStmp.Format("\\%03d.avi", videocount++);
	//CString saveName = IsoVideoSaveName + CStmp; 
	//writerhand = cvCreateVideoWriter(saveName, CV_FOURCC('M','P','4','2'), 10, cvSize(vColor[0]->width, vColor[0]->height), 1);

	double maxnum = 999999;
	double maxdis = 100 * 100;
	bool isFirstEndFrame = false;
	bool isFirstActionFrame = false;
	double thresholdFaceToLevel = 2;
	double faceX, faceY; 
	bool isFaceDetect = false;
	for(int i = 0; i < numFrames; i++)
	{		
		if (!isFaceDetect)
		{
			cvtConToIsoPreProcessFace.OutheadDetectionVIPLSDK(vColor[i], faceTxtFilename, i);
			faceY = cvtConToIsoPreProcessFace.faceY;
			faceX = cvtConToIsoPreProcessFace.faceX;
			if(faceX !=0&& faceY !=0)//detect the face
			{
				isFaceDetect = true;
			}
		}
		if (maphand.find(i) != maphand.end())
		{			
			handposition = maphand.find(i)->second;
			level = gethandPosition(handposition, 0) - threshold;
			double facehigh = 240 - faceY;
			double levelhigh = 240 - level;
			if(facehigh / levelhigh < thresholdFaceToLevel)//This means the level is so closed to the face, the level is wrong.An image's height is 240
				continue;
			else
				break;
		}
		else                                                       //the level is assigned 240 - 5 if the first frame does not detect the hand
		{
			level = 240 - threshold / 2;
			double facehigh = 240 - faceY;
			double levelhigh = 240 - level;
			if(facehigh / levelhigh < thresholdFaceToLevel)//This means the level is so closed to the face, the level is wrong.An image's height is 240
				continue;
			else
				break;
		}

	}



	//test
	//level = 240;

	
	isFirstActionFrame = true;//开始默认是动作结束状态
	isFirstEndFrame = false;
	IsoVideoInfo oneIsoVideoInfo;
	int threshFrame = 5;
	double removedis = 0;
	bool IsFirstDetectHand = true;
	TwoDimension lefthandcenter;
	TwoDimension righthandcenter;
	for(int i = 0; i < numFrames; i++)
	{
		
		//cvLine(vColor[i], cvPoint(1, level), cvPoint(310, level), CV_RGB(255,0,0),1);

		double lowest = 0;
		if (maphand.find(i) == maphand.end())
		{
			lowest = maxnum;
		}
		else
		{
			handposition = maphand.find(i)->second;
			if(IsFirstDetectHand)
			{
				IsFirstDetectHand = false;
				TwoDimension A, B;
				A.x = (handposition.handA1.x + handposition.handA2.x) / 2;
				A.y = (handposition.handA1.y + handposition.handA2.y) / 2;
				B.x = (handposition.handB1.x + handposition.handB2.x) / 2;
				B.y = (handposition.handB1.y + handposition.handB2.y) / 2;
				if(handposition.isDetectOneHand)
				{
					if(A.x < faceX)//A is right hand
					{
						righthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						righthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						lefthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						lefthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
						swapInt(maphand.find(i)->second.handB1.x, maphand.find(i)->second.handA1.x);
						swapInt(maphand.find(i)->second.handB1.y, maphand.find(i)->second.handA1.y);
						swapInt(maphand.find(i)->second.handB2.x, maphand.find(i)->second.handA2.x);
						swapInt(maphand.find(i)->second.handB2.y, maphand.find(i)->second.handA2.y);
					}
					else//A is left hand
					{
						lefthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						lefthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						righthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						righthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
					}
				}
				else 
				{
					if(A.x < B.x)//A is right and B is left hand
					{
						righthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						righthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						lefthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						lefthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
						swapInt(maphand.find(i)->second.handB1.x, maphand.find(i)->second.handA1.x);
						swapInt(maphand.find(i)->second.handB1.y, maphand.find(i)->second.handA1.y);
						swapInt(maphand.find(i)->second.handB2.x, maphand.find(i)->second.handA2.x);
						swapInt(maphand.find(i)->second.handB2.y, maphand.find(i)->second.handA2.y);
					}
					else
					{
						lefthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						lefthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						righthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						righthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
					}
				}
			}
			else
			{
				TwoDimension A, B;
				A.x = (handposition.handA1.x + handposition.handA2.x) / 2;
				A.y = (handposition.handA1.y + handposition.handA2.y) / 2;
				B.x = (handposition.handB1.x + handposition.handB2.x) / 2;
				B.y = (handposition.handB1.y + handposition.handB2.y) / 2;
				double disLeftToA = getTwoDimDis(lefthandcenter, A);//A is left and B is right hand always
				double disLeftToB = getTwoDimDis(lefthandcenter, B);
				if(disLeftToA > disLeftToB)
				{
					lefthandcenter.x = B.x;
					lefthandcenter.y = B.y;
					righthandcenter.x = A.x;
					righthandcenter.y = A.y;
					//exchange the lef and right hand
					swapInt(maphand.find(i)->second.handB1.x, maphand.find(i)->second.handA1.x);
					swapInt(maphand.find(i)->second.handB1.y, maphand.find(i)->second.handA1.y);
					swapInt(maphand.find(i)->second.handB2.x, maphand.find(i)->second.handA2.x);
					swapInt(maphand.find(i)->second.handB2.y, maphand.find(i)->second.handA2.y);
				}
				else
				{
					lefthandcenter.x = A.x;
					lefthandcenter.y = A.y;
					righthandcenter.x = B.x;
					righthandcenter.y = B.y;
				}
			}
			//TwoDimension A, B;
			//A = handposition.handA1;
			//B = handposition.handA2;
			////cvRectangle(vColor[i], cvPoint(A.x,A.y), cvPoint(B.x, B.y), Scalar(0, 255, 255), 1, 1, 0);  
			//if(handposition.isDetectOneHand == false)
			//{
			//	A = handposition.handB1;
			//	B = handposition.handB2;
			//	cvRectangle(vColor[i], cvPoint(A.x,A.y), cvPoint(B.x, B.y), Scalar(0,255,0), 1, 1, 0);  
			//}
			//smoothing process，解决只检测到一只手，且位于level之下的情况，检测出错，smoothing插值
			if (handposition.isDetectOneHand && i != 0)
			{
				double high = (handposition.handA1.y + handposition.handA2.y) / 2.0;
				if(high > level)
				{
					double thresholdDis = 60 * 60;
					
					frameHandPosition front;
					frameHandPosition next;
					TwoDimension fA, nB;
					int fj,nj;
					for(int j = i - 1; j >= 0;j --)
					{
						if(maphand.find(j) != maphand.end())
						{
							front = maphand.find(j)->second;
							if(!front.isDetectOneHand)
							{
								fA.x = (front.handB1.x + front.handB2.x) / 2;
								fA.y = (front.handB1.y + front.handB2.y) / 2;
								fj = j;
								break;
							}
						}
					}
					
					for(int j = i + 1; j < numFrames; j++)
					{
						if(maphand.find(j) != maphand.end())
						{
							next = maphand.find(j)->second;
							if(!next.isDetectOneHand)
							{
								nB.x = (next.handB1.x + next.handB2.x) / 2;
								nB.y = (next.handB1.y + next.handB2.y) / 2;
								nj = j;
								break;
							}
						}
					}
					if(fj != -1&&nj != numFrames)//不是最前面和最后面的
					{
						double twoFrameHandDistance = (fA.x - nB.x) * (fA.x - nB.x) + (fA.y - nB.y) * (fA.y - nB.y);

						if (twoFrameHandDistance < thresholdDis)
						{
							handposition.isDetectOneHand = false;
							handposition.handB1.x = (front.handB1.x + next.handB1.x) / 2;
							handposition.handB1.y = (front.handB1.y + next.handB1.y) / 2; 
							handposition.handB2.y = (front.handB2.y + next.handB2.y) / 2; 
							handposition.handB2.x = (front.handB2.x + next.handB2.x) / 2;
							maphand.find(i)->second.isDetectOneHand = false;
							maphand.find(i)->second.handB1.x = (front.handB1.x + next.handB1.x) / 2;
							maphand.find(i)->second.handB1.y = (front.handB1.y + next.handB1.y) / 2; 
							maphand.find(i)->second.handB2.y = (front.handB2.y + next.handB2.y) / 2; 
							maphand.find(i)->second.handB2.x = (front.handB2.x + next.handB2.x) / 2;
						}
					}
				}
			}
			if(!handposition.isDetectOneHand && i != 0)//矫正错误检测，利用插值拟合
			{
				double tmphighest = gethandPosition(handposition, i);
				if(tmphighest > level)
				{
					int fj, nj;
					frameHandPosition front;
					frameHandPosition next;
					
					for(fj = i - 1; fj >= 0;fj --)
					{
						if(maphand.find(fj) != maphand.end())
						{		
							if(!maphand.find(fj)->second.isDetectOneHand)
							{
								front = maphand.find(fj)->second;
								break;
							}	
						}
					}
					for(nj = i + 1; nj < numFrames;nj ++)
					{
						if(maphand.find(nj) != maphand.end())
						{		
							if(!maphand.find(nj)->second.isDetectOneHand)
							{
								next = maphand.find(nj)->second;
								break;
							}								
						}
					}
					if(fj != -1 && nj != numFrames)
					{
						TwoDimension c11, c12, c21, c22, c31, c32;
						c11.x = (front.handA1.x + front.handA2.x) / 2;
						c11.y = (front.handA1.y + front.handA2.y) / 2;
						c12.x = (front.handB1.x + front.handB2.x) / 2;
						c12.y = (front.handB1.y + front.handB2.y) / 2;

						c21.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						c21.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						c22.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						c22.y = (handposition.handB1.y + handposition.handB2.y) / 2;

						c31.x = (next.handA1.x + next.handA2.x) / 2;
						c31.y = (next.handA1.y + next.handA2.y) / 2;
						c32.x = (next.handB1.x + next.handB2.x) / 2;
						c32.y = (next.handB1.y + next.handB2.y) / 2;


						TwoDimension ave1, ave2;
						ave1.x = (c11.x + c31.x) / 2;
						ave1.y = (c11.y + c31.y) / 2;
						ave2.x = (c12.x + c32.x) / 2;
						ave2.y = (c12.y + c32.y) / 2;
						double twohandMoveDis = getTwoDimDis(ave1, c21) + getTwoDimDis(ave2, c22);
						if(twohandMoveDis > maxdis)
						{
							handposition.isDetectOneHand = false;
							handposition.handB1.x = (front.handB1.x + next.handB1.x) / 2;
							handposition.handB1.y = (front.handB1.y + next.handB1.y) / 2; 
							handposition.handB2.y = (front.handB2.y + next.handB2.y) / 2; 
							handposition.handB2.x = (front.handB2.x + next.handB2.x) / 2;
						}
					}
				}
			}
			lowest = gethandPosition(handposition, i);
		}
		if (lowest < level)
		{
			if(isFirstActionFrame == true)
			{
				std::cout<<"begin frame="<<i<<endl;
				gestureStart = i;
				oneIsoVideoInfo.start = gestureStart;
				
				isFirstActionFrame = false;					//之后就不是第一次动作开始帧了
				isFirstEndFrame = true;
				//CStmp.Format("%03d.avi", videocount++);
				//CString saveName = IsoVideoSaveName + "\\" + CStmp; 
				//writer = cvCreateVideoWriter(saveName, CV_FOURCC('M','P','4','2'), 10, cvSize(vColor[0]->width, vColor[0]->height), 1);
				//cvWriteFrame(writer, vColor[i]);
			}
			else
			{
				//cvWriteFrame(writer, vColor[i]);
			}
		}
		else
		{
			if(isFirstEndFrame == false)
			{
				continue;
			}
			else                                             //FirstEndFrame
			{
				std::cout<<"end frame="<<i<<endl;
				gestureEnd = i;
				oneIsoVideoInfo.end = gestureEnd;
				if(gestureEnd - gestureStart > threshFrame)
				{
					of<<gestureStart<<"	"<<gestureEnd<<endl;
					cvtIsovideoInfo.push_back(oneIsoVideoInfo);
				}
				isFirstActionFrame = true;
				isFirstEndFrame = false;					//之后就不是第一次动作结束帧了

			}
		}

	}
	if(isFirstActionFrame == false)							//if the action don't stop until ending
	{
		cvReleaseVideoWriter(&writer);
		gestureEnd = numFrames - 1;
		oneIsoVideoInfo.end = gestureEnd;
		if(gestureEnd - gestureStart > threshFrame)
		{
			of<<gestureStart<<"	"<<gestureEnd<<endl;
			cvtIsovideoInfo.push_back(oneIsoVideoInfo);
		}
	}
	int numIsoVideo = cvtIsovideoInfo.size();
	int countIsoVideo = 1;
	for(int i = 0; i < numIsoVideo; i++)
	{
		int videostart = cvtIsovideoInfo[i].start;
		int videoend = cvtIsovideoInfo[i].end;
		CStmp.Format("%03d.avi", countIsoVideo++);
		CString saveName = IsoVideoSaveName + "\\" + CStmp; 
		writer = cvCreateVideoWriter(saveName, CV_FOURCC('M','P','4','2'), 10, cvSize(vColor[0]->width, vColor[0]->height), 1);
		for(int j = videostart; j < videoend; j++)
		{
			cvWriteFrame(writer, vColor[j]);
		}
		cvReleaseVideoWriter(&writer);
	}

	//close file
	of.close();
	//delete maphand
	map<int, frameHandPosition>::iterator it;
	for(it = maphand.begin(); it != maphand.end();)
	{
		maphand.erase(it++);
	}
	//release video vector memory
	for(int i = numFrames - 1; i >= 0; i--)
	{
		cvReleaseImage(&vColor[i]);
		vColor.pop_back();
	}
	return 0;
}
double CvtConvideoToIsovideo_flag(CString handLabelFilePath, CString rgbFilePath, CString IsoVideoSaveName, vector<IsoVideoInfo>& cvtIsovideoInfo, CString SegInfoDevelpath)
{
	CString videoname = rgbFilePath.Right(11).Left(5);
	IsoVideoSaveName = IsoVideoSaveName + "\\" + videoname;

	_mkdir(IsoVideoSaveName);

	CString IsoinfoTxt = SegInfoDevelpath + "\\" + videoname + ".txt";
	ofstream of;
	of.open(IsoinfoTxt, ios::app);

	std::cout<<videoname<<endl;
	fstream filestream;

	map <int, frameHandPosition> maphand;
	
	frameHandPosition handposition;
	vector<IplImage*> vColor;//vColor 存储RGB video
	char buffer[256];

	filestream.open(handLabelFilePath, ios::in);
	if(!filestream)
	{
		std::cout<<handLabelFilePath<<" does not exists"<<endl;
		return -1;
	}
	//read handLabelFilePath
	while(filestream.getline(buffer, 256))
	{
		int bufferlen = strlen(buffer);
		{
			int flag1,flag2;
			sscanf(buffer, "%04d %03d %03d %03d %03d %d %03d %03d %03d %03d %d\n", &handposition.nframe, &handposition.handA1.x, &handposition.handA1.y, &handposition.handA2.x, &handposition.handA2.y,&flag1,
				&handposition.handB1.x, &handposition.handB1.y, &handposition.handB2.x, &handposition.handB2.y, &flag2);
		}		
		if(handposition.handB1.x == 0)
			handposition.isDetectOneHand = true;
		else
			handposition.isDetectOneHand = false;
		maphand[handposition.nframe] = handposition;
	}
	// read video
	CvCapture *capture = cvCreateFileCapture(rgbFilePath);
  	int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
	if(numFrames < 1)
		return -1;
	for(int i = 0; i < numFrames; i++)
	{
		IplImage* tempFrame = cvQueryFrame(capture);
		vColor.push_back(cvCloneImage(tempFrame));
	}
	cvReleaseCapture(&capture);

	int gestureStart = 0;//the frame indicates the one gesture begins
	int gestureEnd = 0;//the frame indicates the one gesture ends
	double level = 0;//the level can judge if the gesture begins 
	//IplImage* tempImg = cvCreateImage(cvSize(vColor[0]->width,vColor[0]->height),vColor[0]->depth,vColor[0]->nChannels);
	double threshold = 10;
	int videocount = 0;
	CvVideoWriter *writer = 0;

	CvVideoWriter *writerhand = 0;
	CString CStmp;
	//CStmp.Format("\\%03d.avi", videocount++);
	//CString saveName = IsoVideoSaveName + CStmp; 
	//writerhand = cvCreateVideoWriter(saveName, CV_FOURCC('M','P','4','2'), 10, cvSize(vColor[0]->width, vColor[0]->height), 1);

	double maxnum = 999999;
	double maxdis = 100 * 100;
	bool isFirstEndFrame = false;
	bool isFirstActionFrame = false;
	double thresholdFaceToLevel = 2;
	double faceX, faceY; 
	bool isFaceDetect = false;
	for(int i = 0; i < numFrames; i++)
	{		
		if (!isFaceDetect)
		{
			cvtConToIsoPreProcessFace.OutheadDetectionVIPLSDK(vColor[i], faceTxtFilename, i);
			faceY = cvtConToIsoPreProcessFace.faceY;
			faceX = cvtConToIsoPreProcessFace.faceX;
			if(faceX !=0&& faceY !=0)//detect the face
			{
				isFaceDetect = true;
			}
		}
		if (maphand.find(i) != maphand.end())
		{			
			handposition = maphand.find(i)->second;
			level = gethandPosition(handposition, 0) - threshold;
			double facehigh = 240 - faceY;
			double levelhigh = 240 - level;
			if(facehigh / levelhigh < thresholdFaceToLevel)//This means the level is so closed to the face, the level is wrong.An image's height is 240
				continue;
			else
				break;
		}
		else                                                       //the level is assigned 240 - 5 if the first frame does not detect the hand
		{
			level = 240 - threshold / 2;
			double facehigh = 240 - faceY;
			double levelhigh = 240 - level;
			if(facehigh / levelhigh < thresholdFaceToLevel)//This means the level is so closed to the face, the level is wrong.An image's height is 240
				continue;
			else
				break;
		}

	}



	//test
	//level = 240;

	
	isFirstActionFrame = true;//开始默认是动作结束状态
	isFirstEndFrame = false;
	IsoVideoInfo oneIsoVideoInfo;
	int threshFrame = 5;
	double removedis = 0;
	bool IsFirstDetectHand = true;
	TwoDimension lefthandcenter;
	TwoDimension righthandcenter;
	for(int i = 0; i < numFrames; i++)
	{
		
		//cvLine(vColor[i], cvPoint(1, level), cvPoint(310, level), CV_RGB(255,0,0),1);

		double lowest = 0;
		if (maphand.find(i) == maphand.end())
		{
			lowest = maxnum;
		}
		else
		{
			handposition = maphand.find(i)->second;
			if(IsFirstDetectHand)
			{
				IsFirstDetectHand = false;
				TwoDimension A, B;
				A.x = (handposition.handA1.x + handposition.handA2.x) / 2;
				A.y = (handposition.handA1.y + handposition.handA2.y) / 2;
				B.x = (handposition.handB1.x + handposition.handB2.x) / 2;
				B.y = (handposition.handB1.y + handposition.handB2.y) / 2;
				if(handposition.isDetectOneHand)
				{
					if(A.x < faceX)//A is right hand
					{
						righthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						righthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						lefthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						lefthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
						swapInt(maphand.find(i)->second.handB1.x, maphand.find(i)->second.handA1.x);
						swapInt(maphand.find(i)->second.handB1.y, maphand.find(i)->second.handA1.y);
						swapInt(maphand.find(i)->second.handB2.x, maphand.find(i)->second.handA2.x);
						swapInt(maphand.find(i)->second.handB2.y, maphand.find(i)->second.handA2.y);
					}
					else//A is left hand
					{
						lefthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						lefthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						righthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						righthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
					}
				}
				else 
				{
					if(A.x < B.x)//A is right and B is left hand
					{
						righthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						righthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						lefthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						lefthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
						swapInt(maphand.find(i)->second.handB1.x, maphand.find(i)->second.handA1.x);
						swapInt(maphand.find(i)->second.handB1.y, maphand.find(i)->second.handA1.y);
						swapInt(maphand.find(i)->second.handB2.x, maphand.find(i)->second.handA2.x);
						swapInt(maphand.find(i)->second.handB2.y, maphand.find(i)->second.handA2.y);
					}
					else
					{
						lefthandcenter.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						lefthandcenter.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						righthandcenter.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						righthandcenter.y = (handposition.handB1.y + handposition.handB2.y) / 2;
					}
				}
			}
			else
			{
				TwoDimension A, B;
				A.x = (handposition.handA1.x + handposition.handA2.x) / 2;
				A.y = (handposition.handA1.y + handposition.handA2.y) / 2;
				B.x = (handposition.handB1.x + handposition.handB2.x) / 2;
				B.y = (handposition.handB1.y + handposition.handB2.y) / 2;
				double disLeftToA = getTwoDimDis(lefthandcenter, A);//A is left and B is right hand always
				double disLeftToB = getTwoDimDis(lefthandcenter, B);
				if(disLeftToA > disLeftToB)
				{
					lefthandcenter.x = B.x;
					lefthandcenter.y = B.y;
					righthandcenter.x = A.x;
					righthandcenter.y = A.y;
					//exchange the lef and right hand
					swapInt(maphand.find(i)->second.handB1.x, maphand.find(i)->second.handA1.x);
					swapInt(maphand.find(i)->second.handB1.y, maphand.find(i)->second.handA1.y);
					swapInt(maphand.find(i)->second.handB2.x, maphand.find(i)->second.handA2.x);
					swapInt(maphand.find(i)->second.handB2.y, maphand.find(i)->second.handA2.y);
				}
				else
				{
					lefthandcenter.x = A.x;
					lefthandcenter.y = A.y;
					righthandcenter.x = B.x;
					righthandcenter.y = B.y;
				}
			}
			//TwoDimension A, B;
			//A = handposition.handA1;
			//B = handposition.handA2;
			////cvRectangle(vColor[i], cvPoint(A.x,A.y), cvPoint(B.x, B.y), Scalar(0, 255, 255), 1, 1, 0);  
			//if(handposition.isDetectOneHand == false)
			//{
			//	A = handposition.handB1;
			//	B = handposition.handB2;
			//	cvRectangle(vColor[i], cvPoint(A.x,A.y), cvPoint(B.x, B.y), Scalar(0,255,0), 1, 1, 0);  
			//}
			//smoothing process，解决只检测到一只手，且位于level之下的情况，检测出错，smoothing插值
			if (handposition.isDetectOneHand && i != 0)
			{
				double high = (handposition.handA1.y + handposition.handA2.y) / 2.0;
				if(high > level)
				{
					double thresholdDis = 60 * 60;
					
					frameHandPosition front;
					frameHandPosition next;
					TwoDimension fA, nB;
					int fj,nj;
					for(int j = i - 1; j >= 0;j --)
					{
						if(maphand.find(j) != maphand.end())
						{
							front = maphand.find(j)->second;
							if(!front.isDetectOneHand)
							{
								fA.x = (front.handB1.x + front.handB2.x) / 2;
								fA.y = (front.handB1.y + front.handB2.y) / 2;
								fj = j;
								break;
							}
						}
					}
					
					for(int j = i + 1; j < numFrames; j++)
					{
						if(maphand.find(j) != maphand.end())
						{
							next = maphand.find(j)->second;
							if(!next.isDetectOneHand)
							{
								nB.x = (next.handB1.x + next.handB2.x) / 2;
								nB.y = (next.handB1.y + next.handB2.y) / 2;
								nj = j;
								break;
							}
						}
					}
					if(fj != -1&&nj != numFrames)//不是最前面和最后面的
					{
						double twoFrameHandDistance = (fA.x - nB.x) * (fA.x - nB.x) + (fA.y - nB.y) * (fA.y - nB.y);

						if (twoFrameHandDistance < thresholdDis)
						{
							handposition.isDetectOneHand = false;
							handposition.handB1.x = (front.handB1.x + next.handB1.x) / 2;
							handposition.handB1.y = (front.handB1.y + next.handB1.y) / 2; 
							handposition.handB2.y = (front.handB2.y + next.handB2.y) / 2; 
							handposition.handB2.x = (front.handB2.x + next.handB2.x) / 2;
							maphand.find(i)->second.isDetectOneHand = false;
							maphand.find(i)->second.handB1.x = (front.handB1.x + next.handB1.x) / 2;
							maphand.find(i)->second.handB1.y = (front.handB1.y + next.handB1.y) / 2; 
							maphand.find(i)->second.handB2.y = (front.handB2.y + next.handB2.y) / 2; 
							maphand.find(i)->second.handB2.x = (front.handB2.x + next.handB2.x) / 2;
						}
					}
				}
			}
			if(!handposition.isDetectOneHand && i != 0)//矫正错误检测，利用插值拟合
			{
				double tmphighest = gethandPosition(handposition, i);
				if(tmphighest > level)
				{
					int fj, nj;
					frameHandPosition front;
					frameHandPosition next;
					
					for(fj = i - 1; fj >= 0;fj --)
					{
						if(maphand.find(fj) != maphand.end())
						{		
							if(!maphand.find(fj)->second.isDetectOneHand)
							{
								front = maphand.find(fj)->second;
								break;
							}	
						}
					}
					for(nj = i + 1; nj < numFrames;nj ++)
					{
						if(maphand.find(nj) != maphand.end())
						{		
							if(!maphand.find(nj)->second.isDetectOneHand)
							{
								next = maphand.find(nj)->second;
								break;
							}								
						}
					}
					if(fj != -1 && nj != numFrames)
					{
						TwoDimension c11, c12, c21, c22, c31, c32;
						c11.x = (front.handA1.x + front.handA2.x) / 2;
						c11.y = (front.handA1.y + front.handA2.y) / 2;
						c12.x = (front.handB1.x + front.handB2.x) / 2;
						c12.y = (front.handB1.y + front.handB2.y) / 2;

						c21.x = (handposition.handA1.x + handposition.handA2.x) / 2;
						c21.y = (handposition.handA1.y + handposition.handA2.y) / 2;
						c22.x = (handposition.handB1.x + handposition.handB2.x) / 2;
						c22.y = (handposition.handB1.y + handposition.handB2.y) / 2;

						c31.x = (next.handA1.x + next.handA2.x) / 2;
						c31.y = (next.handA1.y + next.handA2.y) / 2;
						c32.x = (next.handB1.x + next.handB2.x) / 2;
						c32.y = (next.handB1.y + next.handB2.y) / 2;


						TwoDimension ave1, ave2;
						ave1.x = (c11.x + c31.x) / 2;
						ave1.y = (c11.y + c31.y) / 2;
						ave2.x = (c12.x + c32.x) / 2;
						ave2.y = (c12.y + c32.y) / 2;
						double twohandMoveDis = getTwoDimDis(ave1, c21) + getTwoDimDis(ave2, c22);
						if(twohandMoveDis > maxdis)
						{
							handposition.isDetectOneHand = false;
							handposition.handB1.x = (front.handB1.x + next.handB1.x) / 2;
							handposition.handB1.y = (front.handB1.y + next.handB1.y) / 2; 
							handposition.handB2.y = (front.handB2.y + next.handB2.y) / 2; 
							handposition.handB2.x = (front.handB2.x + next.handB2.x) / 2;
						}
					}
				}
			}
			lowest = gethandPosition(handposition, i);
		}
		if (lowest < level)
		{
			if(isFirstActionFrame == true)
			{
				std::cout<<"begin frame="<<i<<endl;
				gestureStart = i;
				oneIsoVideoInfo.start = gestureStart;
				
				isFirstActionFrame = false;					//之后就不是第一次动作开始帧了
				isFirstEndFrame = true;
				//CStmp.Format("%03d.avi", videocount++);
				//CString saveName = IsoVideoSaveName + "\\" + CStmp; 
				//writer = cvCreateVideoWriter(saveName, CV_FOURCC('M','P','4','2'), 10, cvSize(vColor[0]->width, vColor[0]->height), 1);
				//cvWriteFrame(writer, vColor[i]);
			}
			else
			{
				//cvWriteFrame(writer, vColor[i]);
			}
		}
		else
		{
			if(isFirstEndFrame == false)
			{
				continue;
			}
			else                                             //FirstEndFrame
			{
				std::cout<<"end frame="<<i<<endl;
				gestureEnd = i;
				oneIsoVideoInfo.end = gestureEnd;
				if(gestureEnd - gestureStart > threshFrame)
				{
					of<<gestureStart<<"	"<<gestureEnd<<endl;
					cvtIsovideoInfo.push_back(oneIsoVideoInfo);
				}
				isFirstActionFrame = true;
				isFirstEndFrame = false;					//之后就不是第一次动作结束帧了

			}
		}

	}
	if(isFirstActionFrame == false)							//if the action don't stop until ending
	{
		cvReleaseVideoWriter(&writer);
		gestureEnd = numFrames - 1;
		oneIsoVideoInfo.end = gestureEnd;
		if(gestureEnd - gestureStart > threshFrame)
		{
			of<<gestureStart<<"	"<<gestureEnd<<endl;
			cvtIsovideoInfo.push_back(oneIsoVideoInfo);
		}
	}
	int numIsoVideo = cvtIsovideoInfo.size();
	int countIsoVideo = 1;
	for(int i = 0; i < numIsoVideo; i++)
	{
		int videostart = cvtIsovideoInfo[i].start;
		int videoend = cvtIsovideoInfo[i].end;
		CStmp.Format("%03d.avi", countIsoVideo++);
		CString saveName = IsoVideoSaveName + "\\" + CStmp; 
		writer = cvCreateVideoWriter(saveName, CV_FOURCC('M','P','4','2'), 10, cvSize(vColor[0]->width, vColor[0]->height), 1);
		for(int j = videostart; j < videoend; j++)
		{
			cvWriteFrame(writer, vColor[j]);
		}
		cvReleaseVideoWriter(&writer);
	}

	//close file
	of.close();
	//delete maphand
	map<int, frameHandPosition>::iterator it;
	for(it = maphand.begin(); it != maphand.end();)
	{
		maphand.erase(it++);
	}
	//release video vector memory
	for(int i = numFrames - 1; i >= 0; i--)
	{
		cvReleaseImage(&vColor[i]);
		vColor.pop_back();
	}
	return 0;
}

void getFiles( string path, vector<string>& files )  
{  
    //文件句柄  
    long   hFile   =   0;  
    //文件信息  
    struct _finddata_t fileinfo;  
    string p;  
    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)  
    {  
        do  
        {  
            //如果是目录,迭代之  
            //如果不是,加入列表  
            if((fileinfo.attrib &  _A_SUBDIR))  
            {  
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
					 getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
            }  
            else  
            {  
                files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
            }  
        }while(_findnext(hFile, &fileinfo)  == 0);  
        _findclose(hFile);  
    }  
}  

/*
input:src the original image(RGB or grey),dst is MaxConnectedDomain image
function:The function can get a image max connected domain
*/
void GetMaxConnectedDomain(IplImage *src, IplImage *dst)
{
    //IplImage *src = cvLoadImage("wind.png", CV_LOAD_IMAGE_COLOR);  
    cvNamedWindow("原始图像");  
    cvShowImage("原始图像", src);  
	cvWaitKey(100);
    //IplImage*    
	if(src->nChannels > 1)
		cvCvtColor(src, dst, CV_BGR2GRAY);  
    cvNamedWindow("灰度图像");  
    cvShowImage("灰度图像", dst);  
	cvWaitKey(100);
    cvThreshold(dst, dst, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);//OTSU二值化     
    IplConvKernel *element = cvCreateStructuringElementEx(5, 5, 0, 0, CV_SHAPE_ELLIPSE);  
    cvMorphologyEx(dst, dst, NULL, element, CV_MOP_OPEN);//开运算，去除比结构元素小的点     
    cvReleaseStructuringElement(&element);  
    cvNamedWindow("二值图像");  
    cvShowImage("二值图像", dst);  
	cvWaitKey(100);
    int w,h;  
    CvSize sz = cvGetSize(dst);  
  
    int color = 254; // 不对0计数,不可能为255,所以254     
    for (w = 0; w < sz.width; w++)    
    {    
        for (h = 0; h < sz.height; h++)    
        {    
            if (color > 0)    
            {    
                if (CV_IMAGE_ELEM(dst, unsigned char, h, w) == 255)    
                {    
                    // 把连通域标记上颜色     
                    cvFloodFill(dst, cvPoint(w, h), CV_RGB(color, color, color));  
                    color--;  
                }    
            }    
        }    
    }    
    cvNamedWindow("标记颜色后的图像");  
    cvShowImage("标记颜色后的图像", dst);  
	cvWaitKey(100);
    int colorsum[255] = {0};  
    for (w=0; w<sz.width; w++)    
    {    
        for (h=0; h<sz.height; h++)    
        {    
            if (CV_IMAGE_ELEM(dst, unsigned char, h, w) > 0)    
            {    
                colorsum[CV_IMAGE_ELEM(dst, unsigned char, h, w)]++;//统计每种颜色的数量     
            }    
        }    
    }    
    std::vector<int> v1(colorsum, colorsum+255);//用数组初始化vector     
    //求出最多数量的染色，注意max_element的使用方法     
    int maxcolorsum = max_element(v1.begin(), v1.end()) - v1.begin();  
    printf("%d\n",maxcolorsum);  
  
    for (w=0; w<sz.width; w++)    
    {    
        for (h=0; h<sz.height; h++)    
        {    
            if (CV_IMAGE_ELEM(dst, unsigned char, h, w) == maxcolorsum)    
            {    
                CV_IMAGE_ELEM(dst, unsigned char, h, w) = 255;  
            }    
            else    
            {    
                CV_IMAGE_ELEM(dst, unsigned char, h, w) = 0;  
            }    
        }    
    }    
    cvNamedWindow("最大连通域图");  
    cvShowImage("最大连通域图", dst);  
	cvWaitKey(100);  
    cvDestroyAllWindows();  
    return ;
}

bool FCvtVideoToImage(CString vFileNmae, vector<IplImage *>& vectorImage)
{
	CvCapture *vcapture = cvCreateFileCapture(vFileNmae);
	int nframe  = (int)cvGetCaptureProperty(vcapture, CV_CAP_PROP_FRAME_COUNT);
	if(nframe < 1)
	{
		std::cout<<"Reading Video Fail"<<endl;
		return false;
	}
	for(int i = 0; i < nframe; i++)
	{
		IplImage * tmpFrame = cvQueryFrame(vcapture);
		vectorImage.push_back(cvCloneImage(tmpFrame));//此处申请内存
	}
	cvReleaseCapture(&vcapture);
	return true;
}
bool ExtractHogFeature_flag(CString colorPath, CString LabelPath,int label, CString savePath)
{
	vector<IplImage*> vColor;
	fstream labelFile;
	labelFile.open(LabelPath,ios::in);
	if(!labelFile)
		return false;

	CvCapture *capture=cvCreateFileCapture(colorPath);
	
	int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
	if(numFrames < 1)
		return false;
	//cout<<"numFrames:"<<numFrames<<endl;
	for(int i=0;i<numFrames;i++)
	{
		IplImage* tempFrame=cvQueryFrame(capture);
		//IplImage* tempFrameNew=cvCreateImage(cvSize(tempFrame->width,tempFrame->height),tempFrame->depth,tempFrame->nChannels);
		//cvSetImageROI(tempFrame,cvRect(25,20,tempFrame->width-45,tempFrame->height-40));
		//cvResize(tempFrame,tempFrameNew);
		//cvResetImageROI(tempFrame);
		//cvReleaseImage(&tempFrame);
		vColor.push_back(cvCloneImage(tempFrame));
	}

	cvReleaseCapture(&capture);

	fstream hogFile;
	hogFile.open(savePath,ios::out|ios::trunc);

	hogFile<<numFrames<<"\t"<<label<<endl;

	string temp;
	while(getline(labelFile,temp))
	{
		std::stringstream coord(temp);
		int num;
		coord>>num;
		while(!coord.eof())
		{
			int x1,y1,x2,y2, flag;
			coord>>x1>>y1>>x2>>y2>>flag;
			if(x1 >= x2 || y1 >= y2)
				continue;
			cvSetImageROI(vColor[num],cvRect(x1,y1,x2-x1+1,y2-y1+1));
			IplImage* handImage;
			handImage=cvCreateImage(cvSize(64,64),8,3);
			cvResize(vColor[num],handImage);
			cvResetImageROI(vColor[num]);
			IplImage* grayImage;
			grayImage=cvCreateImage(cvSize(64,64),8,1);
			cvCvtColor(handImage,grayImage,CV_RGB2GRAY);

			HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(32,32),cvSize(16,16),cvSize(16,16),9); 

			Mat handMat(grayImage);

			vector<float> descriptors;

			hog->compute(handMat, descriptors,Size(0,0), Size(0,0));

			hogFile<<num<<"\t"<<x1<<"\t"<<y1<<"\t"<<x2<<"\t"<<y2<<"\t"<<flag<<"\t";
			double total = 0;
			for(int i=0;i<descriptors.size();i++)
				hogFile<<descriptors[i]<<"\t";
			hogFile<<endl;

			delete hog;
			cvReleaseImage(&handImage);
			cvReleaseImage(&grayImage);
			descriptors.clear();

			//Mat src2(vColor[num]);
			//cv::rectangle(src2, cv::Rect(x1,y1,x2-x1+1,y2-y1+1), cv::Scalar(255, 0, 0), 3);
		}
	}

	labelFile.close();
	hogFile.close();
	int vsize = vColor.size();
	for(int i = vsize - 1; i >= 0; i--)
	{
		cvReleaseImage(&vColor[i]);
		vColor.pop_back();
	}
	vColor.clear();
	return true;
}
bool ExtractHogFeature(CString colorPath, CString LabelPath,int label, CString savePath)
{
	vector<IplImage*> vColor;
	fstream labelFile;
	labelFile.open(LabelPath,ios::in);
	if(!labelFile)
		return false;

	CvCapture *capture=cvCreateFileCapture(colorPath);
	
	int numFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT); 
	if(numFrames < 1)
		return false;
	//cout<<"numFrames:"<<numFrames<<endl;
	for(int i=0;i<numFrames;i++)
	{
		IplImage* tempFrame=cvQueryFrame(capture);
		//IplImage* tempFrameNew=cvCreateImage(cvSize(tempFrame->width,tempFrame->height),tempFrame->depth,tempFrame->nChannels);
		//cvSetImageROI(tempFrame,cvRect(25,20,tempFrame->width-45,tempFrame->height-40));
		//cvResize(tempFrame,tempFrameNew);
		//cvResetImageROI(tempFrame);
		//cvReleaseImage(&tempFrame);
		vColor.push_back(cvCloneImage(tempFrame));
	}

	cvReleaseCapture(&capture);

	fstream hogFile;
	hogFile.open(savePath,ios::out|ios::trunc);

	hogFile<<numFrames<<"\t"<<label<<endl;

	string temp;
	while(getline(labelFile,temp))
	{
		std::stringstream coord(temp);
		int num;
		coord>>num;
		while(!coord.eof())
		{
			int x1,y1,x2,y2,flag;
			coord>>x1>>y1>>x2>>y2>>flag;
			if(x1 >= x2 || y1 >= y2 || (x1+x2+y1+y2)==0 || flag==1)
				continue;
			cvSetImageROI(vColor[num],cvRect(x1,y1,x2-x1+1,y2-y1+1));
			IplImage* handImage;
			handImage=cvCreateImage(cvSize(64,64),8,3);
			cvResize(vColor[num],handImage);
			cvResetImageROI(vColor[num]);
			IplImage* grayImage;
			grayImage=cvCreateImage(cvSize(64,64),8,1);
			cvCvtColor(handImage,grayImage,CV_RGB2GRAY);

			HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(32,32),cvSize(16,16),cvSize(16,16),9); 

			Mat handMat(grayImage);

			vector<float> descriptors;

			hog->compute(handMat, descriptors,Size(0,0), Size(0,0));

			hogFile<<num<<"\t"<<x1<<"\t"<<y1<<"\t"<<x2<<"\t"<<y2<<"\t";
			double total = 0;
			for(int i=0;i<descriptors.size();i++)
				hogFile<<descriptors[i]<<"\t";
			hogFile<<endl;

			delete hog;
			cvReleaseImage(&handImage);
			cvReleaseImage(&grayImage);
			descriptors.clear();

			//Mat src2(vColor[num]);
			//cv::rectangle(src2, cv::Rect(x1,y1,x2-x1+1,y2-y1+1), cv::Scalar(255, 0, 0), 3);
		}
	}

	labelFile.close();
	hogFile.close();
	int vsize = vColor.size();
	for(int i = vsize - 1; i >= 0; i--)
	{
		cvReleaseImage(&vColor[i]);
		vColor.pop_back();
	}
	vColor.clear();
	return true;
}
//coordination map from RGB to depth video 

void fgenerateMap()
{
    double R1[3][3],R2[3][3],P1[3][4],P2[3][4];
	int width = 320;
	int height = 240;
    CvSize imageSize={width, height};
	//师姐计算出来的对齐参数
	double M1[3][3]={597.27, 0, 322.67, 0, 597.04, 232.64, 0, 0, 1};
    double M2[3][3]={529.45, 0, 319.35, 0, 530.51, 234.79, 0,0, 1};
        
    /*double D1[5]={3.71403933e-01, -1.29047451e+01, 0, 0, 1.56843994e+02};
    double D2[5]={4.39538926e-01, -1.70257187e+01, 0, 0, 2.23054718e+02};*/

	double D1[5]={0, 0, 0, 0, 0};
	double D2[5]={0, 0, 0, 0, 0};
    //double D2[5]={0.1884,-0.4865,0,-0.003,0.0007};

    double R[3][3]={0.9999,-0.0106,    0.0049,  0.0106,   0.9999,   0.0034, -0.0049,   -0.0034 ,  0.9999};
    double T[3]={25.0479, 0.2849, -2.0667};
    CvMat CvM1=cvMat(3,3,CV_64F,M1);
    CvMat CvM2=cvMat(3,3,CV_64F,M2);
    CvMat _D1=cvMat(1,5,CV_64F,D1);
    CvMat _D2=cvMat(1,5,CV_64F,D2);
    CvMat _R=cvMat(3,3,CV_64F,R);
    CvMat _T=cvMat(3,1,CV_64F,T);
    CvMat _R1=cvMat(3,3,CV_64F,R1);
    CvMat _R2=cvMat(3,3,CV_64F,R2);
    CvMat _P1=cvMat(3,4,CV_64F,P1);
    CvMat _P2=cvMat(3,4,CV_64F,P2);

    //imageSize=cvGetSize(bgrFrameL);
    cvStereoRectify(&CvM1,&CvM2,&_D1,&_D2,imageSize,&_R,&_T,&_R1,&_R2,&_P1,&_P2,0,0);

    mx1=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    my1=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    mx2=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    my2=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    cvInitUndistortRectifyMap(&CvM1,&_D1,&_R1,&_P1,mx1,my1);
    cvInitUndistortRectifyMap(&CvM2,&_D2,&_R2,&_P2,mx2,my2);
}

void RGB2DepthRectify(TwoDimension src, TwoDimension &dst)
{
	TwoDimension facemiddle;
	int height = 240;
	int width = 320;
	facemiddle.x = cvmGet(mx1, src.y, src.x);
	facemiddle.y = cvmGet(my1, src.y, src.x);
	//cout<<"hello\n";
	//cout<<src.x<<" "<<cvmGet(mx1, src.y, src.x)<<endl;
	int aroundWidth = 30;
	int beginx = max(0, facemiddle.x - aroundWidth);
	int endx = min(width, facemiddle.x + aroundWidth);
	int beginy = max(0, facemiddle.y - aroundWidth);
	int endy = min(height, facemiddle.y + aroundWidth);
	double mindistance = 99999999999;
	//from depth to 
	for(int i = beginy; i < endy; i++)
	{
		for(int j = beginx; j < endx; j++)
		{
			TwoDimension tmp;
			tmp.x = cvmGet(mx2, i, j);
			tmp.y = cvmGet(my2, i, j);
			double tmpdis = getTwoDimDis(tmp, facemiddle);
			if(mindistance > tmpdis)
			{
				mindistance = tmpdis;
				dst.x = j;
				dst.y = i;
			}
		}
	}
}
void Depth2RGBRectify(TwoDimension src, TwoDimension &dst)
{
	TwoDimension facemiddle;
	int height = 240;
	int width = 320;
	facemiddle.x = cvmGet(mx2, src.y, src.x);
	facemiddle.y = cvmGet(my2, src.y, src.x);
	//cout<<"hello\n";
	//cout<<src.x<<" "<<cvmGet(mx1, src.y, src.x)<<endl;
	int aroundWidth = 30;
	int beginx = max(0, facemiddle.x - aroundWidth);
	int endx = min(width, facemiddle.x + aroundWidth);
	int beginy = max(0, facemiddle.y - aroundWidth);
	int endy = min(height, facemiddle.y + aroundWidth);
	double mindistance = 99999999999;
	//from depth to determination
	for(int i = beginy; i < endy; i++)
	{
		for(int j = beginx; j < endx; j++)
		{
			TwoDimension tmp;
			tmp.x = cvmGet(mx1, i, j);
			tmp.y = cvmGet(my1, i, j);
			double tmpdis = getTwoDimDis(tmp, facemiddle);
			if(mindistance > tmpdis)
			{
				mindistance = tmpdis;
				dst.x = j;
				dst.y = i;
			}
		}
	}
}
void cvtRGBHandLabel2Depth(CString handLabelFilePath, CString LabelSavePath)
{

	fstream filestream;
	ofstream writestream;

	map <int, frameHandPosition> maphand;
	frameHandPosition handposition;
	vector<IplImage*> vColor;//vColor 存储RGB video
	char buffer[256];

	filestream.open(handLabelFilePath, ios::in);
	writestream.open(LabelSavePath, ios::out|ios::trunc);//write
	if(!filestream)
	{
		std::cout<<handLabelFilePath<<" does not exists"<<endl;

	}
	//read handLabelFilePath
	while(filestream.getline(buffer, 256))
	{
		int bufferlen = strlen(buffer);
		if(bufferlen <= 20)//detect one hand
		{
			handposition.isDetectOneHand = true;
			sscanf(buffer, "%04d %03d %03d %03d %03d\n", &handposition.nframe, &handposition.handA1.x, &handposition.handA1.y, &handposition.handA2.x, &handposition.handA2.y);
			TwoDimension tmp;
			RGB2DepthRectify(handposition.handA1, handposition.handA1);
			RGB2DepthRectify(handposition.handA2, handposition.handA2);
			char outcontent[200];
			sprintf(outcontent, "%04d %03d %03d %03d %03d\n", handposition.nframe, handposition.handA1.x, handposition.handA1.y, handposition.handA2.x, handposition.handA2.y);
			//writestream<<handposition.nframe<<" "<<handposition.handA1.x<<" "<<handposition.handA1.y<<" "<<handposition.handA2.x<<" "<<handposition.handA2.y<<endl;
			writestream<<outcontent;
			handposition.handB1.x = 0;
			handposition.handB1.y = 0;
			handposition.handB2.x = 0;
			handposition.handB2.y = 0;
		}
		else//detect two hands
		{
			handposition.isDetectOneHand = false;
			sscanf(buffer, "%04d %03d %03d %03d %03d %03d %03d %03d %03d\n", &handposition.nframe, &handposition.handA1.x, &handposition.handA1.y, &handposition.handA2.x, &handposition.handA2.y,
				&handposition.handB1.x, &handposition.handB1.y, &handposition.handB2.x, &handposition.handB2.y);
			RGB2DepthRectify(handposition.handA1, handposition.handA1);
			RGB2DepthRectify(handposition.handA2, handposition.handA2);
			RGB2DepthRectify(handposition.handB1, handposition.handB1);
			RGB2DepthRectify(handposition.handB2, handposition.handB2);
			char outcontent[200];
			sprintf(outcontent, "%04d %03d %03d %03d %03d %03d %03d %03d %03d\n", handposition.nframe, handposition.handA1.x, handposition.handA1.y, handposition.handA2.x, handposition.handA2.y, handposition.handB1.x, handposition.handB1.y, handposition.handB2.x, handposition.handB2.y);
			writestream<<outcontent;
			//writestream<<handposition.nframe<<" "<<handposition.handA1.x<<" "<<handposition.handA1.y<<" "<<handposition.handA2.x<<" "<<handposition.handA2.y<<" "<<handposition.handB1.x<<" "<<handposition.handB1.y<<" "<<handposition.handB2.x<<" "<<handposition.handB2.y<<endl;
		}
		maphand[handposition.nframe] = handposition;
	}
	filestream.close();
	writestream.close();
}

int getCurrentDir(string path) 
{ 
	vector<string> files;
	//文件句柄 
	long  hFile  =  0; 
	//文件信息 
	struct _finddata_t fileinfo; 
	string p; 
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) != -1) 
	{ 
		do
		{  
			if((fileinfo.attrib & _A_SUBDIR)) 
			{ 
				if(strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0) 
				{
					files.push_back(fileinfo.name);
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
				}

			}  
		}while(_findnext(hFile, &fileinfo) == 0); 
		_findclose(hFile); 
	} 
	return files.size();
} 

void getDefinedFiles(string path, vector<string>& results)
{
	HANDLE hFile;
	LPCTSTR lpFileName = path.c_str();
	WIN32_FIND_DATA pNextInfo;	//搜索得到的文件信息将储存在pNextInfo中;
	hFile = FindFirstFile(lpFileName, &pNextInfo);//请注意是 &pNextInfo , 不是pNextInfo;
	if(hFile == INVALID_HANDLE_VALUE)
	{
		//搜索失败
		exit(-1);
	}
	else
	{
		do 
		{
			if(pNextInfo.cFileName[0] == '.')//过滤.和..
				continue;
			results.push_back(pNextInfo.cFileName);
		} while(FindNextFile(hFile, &pNextInfo));
	}
}

void getCurrentDir(string path, vector<string>& files) 
{ 
	//文件句柄 
	long  hFile  =  0; 
	//文件信息 
	struct _finddata_t fileinfo; 
	string p; 
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) != -1) 
	{ 
		do
		{  
			if((fileinfo.attrib & _A_SUBDIR)) 
			{ 
				if(strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0) 
				{
					files.push_back(fileinfo.name);
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
				}

			}  
		}while(_findnext(hFile, &fileinfo) == 0); 
		_findclose(hFile); 
	} 
} 