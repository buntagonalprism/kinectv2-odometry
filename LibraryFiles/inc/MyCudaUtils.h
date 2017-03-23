


#ifndef CUDA_MAPPING_H
#define CUDA_MAPPING_H

/*****************************************************************************
*					    I N C L U D E    F I L E S
*****************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>

#include <stdio.h>
#include <iostream>
#include <time.h>


/*****************************************************************************
*					      C U D A    M A C R O S
*****************************************************************************/

typedef enum
{
	CCX = 0,
	CCY = 1,
	CFX = 2,
	CFY = 3,
	DCX = 4,
	DCY = 5,
	DFX = 6,
	DFY = 7,
	TRANS = 8,
}CamIdxs;
#define CAMPAMSZ 24

typedef enum
{
	CNX = 0,
	CNY = 1,
	DNX = 2,
	DNY = 3,
}ImIdxs;
#define IMPAMSZ 4


/*****************************************************************************
*					      C U D A    M A C R O S
*****************************************************************************/

#define CUDA_CHECK(func) { cudaError_t cudaStatus = func; if (cudaStatus != cudaSuccess)throw (cudaStatus); }

#define CUDA_INIT_MEM(D_dest, H_src, bytes) { CUDA_CHECK( cudaMalloc((void**)&D_dest, bytes) ); CUDA_CHECK( cudaMemcpy(D_dest, H_src, bytes, cudaMemcpyHostToDevice) ); }

#define CUDA_MALLOC(D_dest, bytes) {CUDA_CHECK(cudaMalloc((void**)&D_dest, bytes));}

#define CUDA_DOWNLOAD(H_dest, D_src, bytes) { CUDA_CHECK(cudaMemcpy(H_dest, D_src, bytes, cudaMemcpyDeviceToHost)); }

#define CUDA_UPLOAD(D_dest, H_src, bytes) { CUDA_CHECK(cudaMemcpy(D_dest, H_src, bytes, cudaMemcpyHostToDevice)); }



/*****************************************************************************
*					    P U B L I C   F U N C T I O N S
*****************************************************************************/
namespace MyCudaUtils {
	

	// Maps a kinect colour image into the depth image space
	// It is currently hard-coded with the intrinsic and extrinsic parameters of the Kinectv2 camera
	// [in]  The source colour image
	// [in]  A synchronised depth image
	// [out] The colour image mapped to the depth image space
	_declspec(dllexport) cudaError_t MapColourToDepth(cv::Mat& cimage_in, cv::Mat& dimage_in, cv::Mat& mapped_out, bool singleBlock = false);



	// Maps a kinect depth image into the colour image space
	// Uses an array of the camera parameters due to incompatability with Eigen transforms in KinectCamParams struct
	// [in]  The source depth image
	// [out] A sparse image, the size of a colour image
	// [in]  Camera parameters defined using the array indices above
	_declspec(dllexport) cudaError_t MapDepthToColour(cv::Mat& dImage, cv::Mat& mImage, float* camPams, int* imSizes);




	// Transform an array of 3D points by a homogenous transform
	// [in]  Points_in - (3xN) array of (x,y,z) points stored row-major order
	// [in]  Transfrom - (4x4) homogenous matrix stored in row-major order
	// [out] Points_out - Output array - requires external allocation
	// [in]  Num_pts - the number of points N in the array to transform
	_declspec(dllexport) cudaError_t TransformPoints(float* points_in, float* transform, float* points_out, int num_pts);


		
	// Parallelised brute-force matching. Every descriptor in new keypoint set is compared
	// against all old descriptors by a single CUDA thread.
	// [in]  The descriptor list for the current frame keypoints
	// [in]  The descriptor list for the previous frame keypoints
	// [out] The index of the old keypoint list of the best and second best match to every new keypoint
	// [out] The Hamming distance the old keypoint is from the new keypoint for best and second best matches. 
	_declspec(dllexport) cudaError_t BruteHammingCUDA(cv::Mat& new_desc, cv::Mat& old_desc, cv::Mat& match_idx, cv::Mat& match_dist);



	// Computes Star Keypoints 
	// [in]  Image to process
	// [out] Vector of keypoint locations
	// [in]  maxSize size of the largest octagonal bi-level filter to apply to the image. Enlarge to detect bigger features
	// [in]  responseThresh minimum threshold response required to be considered a keypoint. Enlarge to choose only stronger features
	// [in]  lineThreshProjected
	// [in]  lineThreshBinarised This and the above eliminate keypoints based upon a strong line response
	// [in]  Size of the window in which non-maximum responses will be suppressed. Enlarge to reduce proximity of features
	_declspec(dllexport) cudaError_t ComputeStarKeypoints(cv::Mat& image, std::vector<cv::KeyPoint>& kps, int _maxSize = 45, int _responseThreshold = 30,
		int _lineThresholdProjected = 10, int _lineThresholdBinarized = 8, int _suppressNonmaxSize = 5);



	// Computes BRIEF descriptor for a set of keypoints 
	// [in]  Image to process
	// [in]  Vector of detected keypoints
	// [out] Matrix containing binary descriptor for each keypoint
	// [in]  Size of the descriptor in bytes, 32bytes = 256bit descriptor. Other sizes not currently supported
	_declspec(dllexport) cudaError_t ComputeBriefDescriptors(cv::Mat& image, std::vector<cv::KeyPoint>& kps, cv::Mat& desc, int descSize = 32);



	// Load data onto GPU before beginning RANSAC
	// Saves 
	_declspec(dllexport) cudaError_t RansacLoadCloud(float* cloud_in, float* kps_new, int num_pts, float* camPams, double min_dist_sqd_in);


	// Launches the kernel for computing the reprojection distance for a given transform
	// Non-blocking to allow other RANSAC estimation functions to proceed concurrently on CPU
	_declspec(dllexport) cudaError_t RansacCalcReprojDist(float* transform);
	
	// Downloads the array of each reprojection pixel distance from the GPU
	_declspec(dllexport) cudaError_t RansacGetDists(double* distArray);

	// Get the count of inliers
	// Pass in a pointer to an array to get the distances downloaded as well
	_declspec(dllexport) cudaError_t RansacGetInlierCount(int* inliers, double* distArray = NULL);

	_declspec(dllexport) void RansacTidy(void);

	// Perform Dense ICP on two depth images from the Kinect
	// [in]  The new and old depth images to align
	// [in]  A hint for the relative transform. This transform should be the pose of old frame with respect to the new frame
	//       This transform will be initially applied to all the old points to bring them into estimated new frame
	_declspec(dllexport) cudaError_t DenseICP(cv::Mat& dimage_new, cv::Mat& dimage_old, float* camPams, int* imSizes, float* T_hint, float depthStats[192]);


	

} // End MyCudaUtils namespace

#endif