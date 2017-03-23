/***************************************************************************************

File Name:		MyRGBD.h
Author:			Alex Bunting
Date Modified:  11/8/14

Description:
Contains utility functions for processing RGBD data to estimate relative transforms between frames

****************************************************************************************/

#ifndef MY_RGBD_H
#define MY_RGBD_H

#include "MySimulatorv2.h"
#include "MyTimer.h"
#include "MyCudaUtils.h"
//#include "MyDrawingUtils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2\core\mat.hpp>

#include <pcl/common/eigen.h>
#include <pcl/common/impl/eigen.hpp>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>

#include <intrin.h>
#include <vector>

typedef Eigen::Matrix<float, 3, Eigen::Dynamic> EigenDynamicCloud;

typedef enum
{
	STD_STAR_BRUTE_MATCH = 0,
	MOD_STAR_BRUTE_MATCH = 1,
} VoMethods;

#define IMG_BUF_SIZE 10

namespace MyRGBD {

	struct RansacParams {
	public:
		RansacParams(int min_sample_size_in = 3, int ransac_iterations_in = 200, double inlier_thresh_in = 2.0, KinectCamParams cam_in = KinectCamParams()) :
			min_sample_size(min_sample_size_in),
			ransac_iterations(ransac_iterations_in),
			inlier_thresh(inlier_thresh_in),
			inlier_thresh_sqd(pow(inlier_thresh_in, 2)),
			cam(cam_in) {}

		int min_sample_size;
		int ransac_iterations;
		double inlier_thresh;
		double inlier_thresh_sqd;
		KinectCamParams cam;
	};

	
	// Finds and describes features in a greyscale image
	// Uses the initialised Star detector and BRIEF descriptor, with default properties if not otherwise defined
	// [in]  Greyscale image to process
	// [in]  Feature detector to call
	// [in]  Feature extractor to call
	// [out] Vector of keypoints found
	// [out] Matrix containing descriptors of the keypoints
	// [in]  Options set including timing variables 
	void FindAndDescribeFeatures(cv::Mat& gimage_in, cv::FeatureDetector& detector, cv::DescriptorExtractor& extractor, std::vector<cv::KeyPoint>& kps_out, cv::Mat& kp_desc_out);


	// Matches features between two sets of computed keypoints
	// [in]  Feature descriptors for current frame
	// [in]  Feature descriptors in previous frame
	// [out] For each keypoint in the current frame, the index of its match in the previous frame
	// [out] For each match, the distance between the descriptors
	void MatchFeatures(cv::Mat& kp_desc_new_in, cv::Mat& kp_desc_old_in, cv::Mat& match_idx_out, cv::Mat& match_dist_out);


	// Performs Brute-Force hamming distance matching between a set of descriptors
	// Accelerated by SSE4 instruction POPCNT
	// An alternative to FLANN-based matching with more efficient computation
	// for small (< 20,000) sets due to SSE acceleration
	void BruteHammingSSE(cv::Mat& new_desc, cv::Mat& old_desc, cv::Mat& match_idx, cv::Mat& match_dist);


	// Filters the matches to select only those which meet Lowe's criteria
	// That the match distance of the best match should be 0.6*second best
	// Additionally filters keypoints close to the Frustum edges
	vector<cv::DMatch> LowesFilterFeatures(cv::Mat& match_idx_in, cv::Mat& match_dist_in, std::vector<cv::KeyPoint>& kps_new,
		std::vector<cv::KeyPoint>& kps_old, cv::Mat& gimage_new, cv::Mat& gimage_old);


	// Projects two keypoint vector sets to point clouds 
	vector<cv::DMatch> ProjectKeypointsToCloud(vector<cv::DMatch>& good_matches, EigenDynamicCloud& keypoint_cloud_new, EigenDynamicCloud& keypoint_cloud_old,
		cv::Mat& mimage_new, cv::Mat& mimage_old, vector<cv::KeyPoint>& new_kp, vector<cv::KeyPoint>& old_kp);



	// Consider reweighting keypoints near border - increases robustness to calibration error according to libviso
	vector<cv::DMatch> RansacFilterFeatures(vector<cv::DMatch>& matches, vector<cv::KeyPoint>& new_kp, vector<cv::KeyPoint>& old_kp, 
		EigenDynamicCloud& kp_cloud_new, EigenDynamicCloud& kp_cloud_old, RansacParams& ranPams = RansacParams());


	void EstimateTransform(EigenDynamicCloud& kp_cloud_new, EigenDynamicCloud& kp_cloud_old, Eigen::Matrix4f& trans_out);


	vector<int> GetInliers(vector<cv::DMatch>& matches, vector<cv::KeyPoint>& new_kp, EigenDynamicCloud& old_cloud, Eigen::Matrix4f& transform, RansacParams& ranPams = RansacParams());


	vector<int> GetRandomSample(int N, int num);

}	// End MyRGBD namespace



#endif

