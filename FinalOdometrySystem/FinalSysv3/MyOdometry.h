/***************************************************************************************

File Name:		MyOdometry.h
Author:			Alex Bunting
Date Modified:  3/9/14

Description:
The definition for the MyOdometry class, which handles experiment data and calls various
subroutines for computing odometry on collected colour, depth and IMMU data. 

****************************************************************************************/

#ifndef MY_ODOMETRY_H
#define MY_ODOMETRY_H

#include "MyRGBD.h"
#include "MyAHRS.h"
#include "MySimulatorv2.h"
#include "MySBA.h"
#include "MyOdometryDatatypes.hpp"
#include "MyCudaUtils.h"
#include "MyDrawingUtils.h"

#include <pcl\gpu\kinfu\kinfu.h>

#include <string>
#include <iostream>

class MyOdometry {

public:


	// Initialise MyOdometryClass
	// [in]  Number of frames to keep in internal buffers
	//       Also number of frames to perform SBA over if used
	MyOdometry(int numframes);


	// Currently empty destructor
	~MyOdometry();


	// Initialise source of data. Internally creates a MySimulator object to fetch the data
	// [in]  The root folder containing all the experiment folders
	// [in]  The name of the experiment folder to load
	// [in]  The types of observations to load - defined in MySimulatorv2.h
	// [in]  Wether the folder contains mirrored or normal images. Mirrored is default
	void InitDataSource(char* inputFolderDir, char* inputFolderName, int obsToLoad = OBS_COLOUR_AND_DEPTH | OBS_IMMU_AND_GPS, ImageMirroring mirroring = IMAGES_MIRRORED);


	// Initialise output to files. 
	// Checks existence and state of requested folders and verifies via command line
	// [in]  The root folder for output
	// [in]  The folder to output to
	void InitOutput(char* outputFolderDir, char* outputFolderName);


	// Initialise the star tracker settings. The default values for each entry are the OpenCV default settings
	// If this function is not called, the default values indicated will be used automatically
	// [in]  maxSize size of the largest octagonal bi-level filter to apply to the image. Enlarge to detect bigger features
	// [in]  responseThresh minimum threshold response required to be considered a keypoint. Enlarge to choose only stronger features
	// [in]  lineThreshProjected
	// [in]  lineThreshBinarised This and the above eliminate keypoints based upon a strong line response
	// [in]  Size of the window in which non-maximum responses will be suppressed. Enlarge to reduce proximity of features
	void InitStarTracker(int maxSize = 45, int responseThresh = 30, int lineThreshProjected = 10, int lineThresBinarised = 8, int suppressNonmaxSize = 5);



	void InitBriefDescriptor(int bytes = 32);



	void InitMatcher(int matchType, float best_ratio = 0.6);



	void InitSBA(void);



	void InitRansac(int min_sample_size_in = 3, int ransac_iterations_in = 200, double inlier_thresh_in = 2.0, KinectCamParams cam_in = KinectCamParams());



	void SetOpts(int timingDetail = 0, int verbosityLevel = 0, int drawing = 0);



	void Start(int framesToProcess);


	// Routine for testing experimental functions during debugging
	void StartExperimental(void);



	void VerifyOutputDir(char* outfolderDir, char* outfolderName);



	void OpenOutputFiles(ofstream& rgbdPosesFile, ofstream& sbaPosesFile, ofstream& poseTimesFile,
		ofstream& ahrsFile, ofstream& ahrsTimes, ofstream& rawImmuOutFile, ofstream& calibImmuOutFile, 
		std::ofstream& keypointCountsFiles, std::ofstream& icpfForwardPosesFile,std::ofstream& icpReversePosesFile,
		std::ofstream& depthImStatsFile, char* outfolderDir, char* outfolderName);

private:
	// Data grabber
	MySimulatorv2 sim;

	// SBA Constants
	MySBA::SBAParams sbaParams;

	// Ransac Paramers
	MyRGBD::RansacParams ranPams;

	// Debugging, timing and visualisation options
	OdoOpts odoOpts;

	// All world points in the frame set
	vector<world_pt> world_pts;
	
	// Counters
	const int numframes;
	int num_meas;
	int frame;

	// Output Files
	std::ofstream rgbdPosesFile;			
	std::ofstream sbaPosesFile;
	std::ofstream poseTimesFile;
	std::ofstream ahrsFile;
	std::ofstream ahrsTimes;
	std::ofstream rawImmuOutFile;
	std::ofstream calibImmuOutFile;
	std::ofstream keypointCountsFiles;
	std::ofstream depthImStatsFile;
	std::ofstream icpfForwardPosesFile;
	std::ofstream icpReversePosesFile;
	

	// Options for the star 
	struct StarOpts {
		int maxSize;
		int responseThresh;
		int lineThreshProjected;
		int lineThresBinarised;
		int suppressNonmaxSize;
	} starOpts;

	// Stores data about each image
	vector<Mat> mimages;						// Mapped depth images
	vector<Mat> gimages;						// Greyscale images
	vector<Mat> dimages;
	vector<frame_kps> kps;						// Keypoints in each frame
	vector<Mat> kp_desc;						// Descriptors
	vector<EigenDynamicCloud> kp_clouds;		// Keypoints projected to 3D clouds

	// Temporary images
	Mat dimage, cimage, dummy;

	// For storing pose estimate chains
	vector<Eigen::Matrix4f> cam_poses;
	vector<Eigen::Matrix4f> ba_poses;

	// Incremental pose estimates
	Eigen::Matrix4f relative_VO_pose, relative_sba_pose;
	Eigen::Matrix4f concat_VO_pose;
	Eigen::Matrix4f last_ba_pose;

	// For feature finding
	cv::StarFeatureDetector featureDetector;
	cv::BriefDescriptorExtractor featureExtractor;
	
	// For keypoint matching
	Mat match_idx, match_dist;
	vector<cv::DMatch> good_matches;
	vector<cv::DMatch> projected_matches;
	vector<cv::DMatch> inlier_matches;
	float best_ratio;

	// AHRS Data
	MyAHRS ahrs;
	int ahrs_cnt;
	std::string immudataline;
	long long timestamp;
	float accel[3], mag[3], gyro[3], rotation[3][3];

	Timer timer;

};


namespace MyCudaOdometry {
	vector<cv::DMatch> RansacFilterFeatures(vector<cv::DMatch>& projected_matches, vector<frame_kps>& kps, vector<EigenDynamicCloud>& kp_clouds,
		int frame, MyRGBD::RansacParams& ranPams, float* camArray);
}

#endif