/***************************************************************************************

File Name:		MySimulatorv2.h
Author:			Alex Bunting
Date Modified:  21/7/14

Description:	
Contains the definition of the MySimulatorv2 class, used for playing back recorded data
from a Kinectv2 sensor. Functions include the ability to play streams as videos and load
a set of observations in timestamped order. 

****************************************************************************************/

#ifndef MY_SIMULATORV2_H
#define MY_SIMULATORV2_H

#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <regex>

#include "boost\filesystem.hpp"

#include "opencv2\highgui\highgui.hpp"
#include "opencv2\contrib\contrib.hpp"

#include "MyKinectv2Utils.h"

typedef enum 
{
 VISUAL_COLOUR		=	1,
 VISUAL_GREY		=	2,
 DEPTH_COLOUR_REL	=	3,
 DEPTH_COLOUR_ABS	=	4,
 DEPTH_GREY_REL		=	5,
 DEPTH_GREY_ABS		=	6,
 POINTS_GREY		=	7,
 POINTS_COLOUR		=	8,
 DEPTH_RAW			=   9,
} StreamTypes;

typedef enum
{
	OBS_END					= 0,
	OBS_COLOUR				= 1,
	OBS_DEPTH				= 2,
	OBS_GPS					= 4,
	OBS_IMMU				= 8,
	OBS_IR					= 16,
	OBS_COLOUR_AND_DEPTH	= 32,
	OBS_IMMU_AND_GPS		= 64,
} ObsTypes;

typedef enum
{
	IMAGES_MIRRORED = 1,	// Images are mirrored - designed to be displayed on a screen to mimic the user
	IMAGES_NORMAL = 0,		// Images are what a human eye at the camera location would see
} ImageMirroring;

typedef std::multimap<long long, boost::filesystem::path>::iterator FileIter;
using namespace MyKinectv2Utils;

class MySimulatorv2 {
public:
	// Empty default constructor
	MySimulatorv2() {};
	
	// Opens and reads a folder of collected data
	// [in]  The root folder containing all the experiment folders
	// [in]  The name of the experiment folder to load
	// [in]  Wether the folder contains mirrored or normal images. Mirrored is default
	void LoadData(char* rootFolder, char* folderToOpen, char obsToLoad, ImageMirroring mirroring = IMAGES_MIRRORED);


	// Destructor closes opened files
	~MySimulatorv2();


	// Sets depth image scaling for greyscale / coloured display
	// [in]  Minimum and maximum values to scale the depth values between
	//       Anthing outside this range is scaled to 0 or 255 respectively
	void SetDepthImageScaling(int mindepth_mm, int maxdepth_mm);


	// Returns a list of all available files in the experiment folder
	// [in]  The type of observation to get the list for
	// [ret] An iterator to all the files in the folder
	FileIter GetFileList(ObsTypes FILE_TYPE);


	// Used to check for the existance of more time-stamped observations within experiment folder
	// [ret] A flag indicating the next observation
	ObsTypes HasMoreObs(void);


	// Gets the next timestamped observation from either colour or depth camera
	// [out] The image in which the observation will be stored
	// [ret] The timestamp of the observation
	long long GetNextObs(cv::Mat& image_out);


	// Gets syncrhonised timestamped observations from both colour and depth cameras
	// [out] The colour image in which the observation will be stored
	// [out] The depth image in which the observation will be stored
	// [ret] The timestamp of the observation
	long long GetNextObs(cv::Mat& colour_image_out, cv::Mat& depth_image_out);


	// Gets the next timestamped observation from either IMU or GPS
	// [out] The string in which the observation will be stored
	// [ret] The timestamp of the observation
	long long GetNextObs(std::string& line);


	// Plays back data from a single captured stream
	// [in]  The stream type to play
	void PlayStream(StreamTypes STREAM);


	// Displays a single image from a stream
	// [in]  The stream type to play
	// [in]  The image number in the stream to play
	void DisplayStreamImage(StreamTypes STREAM, int img_number);


	// Displays a single line of recorded data using its index of the total
	void PrintStreamData(int STREAM, int reading_number);


	// Return a stream image using an index
	void LoadStreamCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int img_number);

	// Return a pair of colour and depth frames at a given index
	void LoadColourAndDepthFrame(int d_frame_idx, cv::Mat& cimage_out, cv::Mat& dimage_out);

private:
	
	ImageMirroring mirroring;

	std::multimap<long long, boost::filesystem::path> depthFileList;		// List of depth files in directory
	std::multimap<long long, boost::filesystem::path> colourFileList;		// List of colour files in the directory

	char folderDir[256];													// Stores root folder for experiments
	char folderName[256];													// Stores experiment folder name to load

	int depthFileCount, colourFileCount;									// Total count of depth files in directory
												
	int minDepthScaling, maxDepthScaling;									// Parameters for scaling depth image appearances

	struct ObsTimes { 
		long long depthTime, colourTime, gpsTime, immuTime, immuGpsTime; 
		ObsTypes GetMinType(void) {
			long long min = colourTime;
			ObsTypes min_type = OBS_COLOUR;
			if (depthTime < min) { min = depthTime; min_type = OBS_DEPTH; }
			if (gpsTime < min) { min = gpsTime; min_type = OBS_GPS; }
			if (immuTime < min) { min = immuTime; min_type = OBS_IMMU; }
			if (immuGpsTime < min) { min = immuGpsTime; min_type = OBS_IMMU_AND_GPS; }
			if (min_type == OBS_COLOUR && depthTime == colourTime)
				min_type = OBS_COLOUR_AND_DEPTH;
			return min_type;
		}
	};																		
	ObsTimes times;															// Stores time of the next observation

	FileIter depthIter, colourIter;											// Iterates through files in folder

	std::ifstream immuFile, gpsFile, immuGpsFile;							// Defaults to input stream only

	char immuFileName[256], gpsFileName[256], immuGpsFileName[256];			// Name of text files to open

	char immuDataLine[256], gpsDataLine[256], immuGpsDataLine[256];			// Stores a stream of loaded data

	ObsTypes nextObs;														// Holds the type of the next observation

	char obsToLoad;															// Holds which observations will be loaded

	// Internal functions for drawing different streams
	void PlayDepth(StreamTypes stream, FileIter start, FileIter end, int delay_ms);
	void PlayVisual(StreamTypes stream, FileIter start, FileIter end, int delay_ms);
	void PlayPoints(StreamTypes stream, FileIter start, FileIter end, int delay_ms);

	void DepthToRelativeIntensity(cv::Mat& depthRaw, cv::Mat& depthImOut);
	void DepthToAbsoluteIntensity(cv::Mat& depthRaw, cv::Mat& depthImOut);

};

#endif