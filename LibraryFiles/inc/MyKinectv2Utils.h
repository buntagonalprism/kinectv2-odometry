/***************************************************************************************

File Name:		MyKinectv2Utils.h
Author:			Alex Bunting
Date Modified:  21/7/14

Description:
Contains various utility functions for manipulating data from a Kinect for Windows v2 
sensor. These are mostly concerned with coordinate transforms between different camera
spaces, projecting images to clouds and reprojecting clouds to images. 

****************************************************************************************/
#ifndef MY_KINECTV2_UTILS
#define MY_KINECTV2_UTILS


// EXTERNAL INCLUDES 
#include "MyKinectv2.h"

// MY INCLUDES

#include <boost/math/special_functions/round.hpp>

#include <opencv2\highgui\highgui.hpp>

// These need to be immediately before any PCL includes
// To remove windows defines of min and max

#ifndef NOMINMAX
#error "PCL Conflicts with the Windows min/max functions. Define NOMINMAX preprocessor directive to remove them"
#endif

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <pcl/common/eigen.h>
#include <pcl/common/impl/eigen.hpp>

// NAMESPACES 
using cv::Mat;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr xyzCloudPtr;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgbCloudPtr;


namespace MyKinectv2Utils {
	
	// Project a depth image to a point cloud using a set of camera intrinsics
	// [in]  The depth image to project, with depths in mm
	// [in]  The intrinsic camera parameters which produced the depth image
	// [out] The output cloud, with pionts in m
	void ProjectDepthImageToCloud(Mat& depthImage_in, xyzCloudPtr cloud_out, float fx, float fy, float cx, float cy);


	// As above but overloaded to take the default kinect instead
	// Assumes projection from the depth camera
	void ProjectDepthImageToCloud(Mat& depthImage_in, xyzCloudPtr cloud_out, const KinectCamParams& cam = KinectCamParams());


	// Projects a colour pixel to an Eigen-style point using default colour camera intrinsics
	// [in]  The depth image mapped to the colour space
	// [in]  The pixel coordinates to project
	// [out] The 3D coordinates of the point
	// [ret] Success or failure (failure occurs if there is no depth data at the pixel location)
	bool ProjectColourPixelToPoint(Mat& depth_mapped_in, int row, int col, Eigen::Matrix < float, 3, 1 >& pt_out, const KinectCamParams& cam = KinectCamParams());


	// Transform a point cloud by a homogenous transform
	// [in]  The input cloud
	// [in]  The 4x4 homogenous transformation matrix to apply
	// [out] The output cloud
	void TransformCloud(xyzCloudPtr cloud_in, xyzCloudPtr cloud_out, Eigen::Matrix4f transform);


	// As above but overloaded to take the defaul kinect instead
	// Assumes transformation from colour to depth camera frames
	void TransformCloud(xyzCloudPtr cloud_in, xyzCloudPtr cloud_out, const KinectCamParams& cam = KinectCamParams());


	// Projects a cloud back to an image using the specified camera intrinsics and image dimensions
	// [in]  Cloud to reproject back to an image
	// [in]  Dimensions of the image to form
	// [in]  Intrinsic camera parameters of the camera observing the image
	// [out] Depth image of the cloud of specified dimensions taken by specified camera
	// [ret] The number of depth points successfully mapped to the colour image 
	int ReprojectCloudToImage(xyzCloudPtr cloud_in, Mat& dimage_out, float fx, float fy, float cx, float cy, int num_rows, int num_cols);


	// As above but overloaded to take the default kinect instead
	// Assumes that reprojection back to the colour image is desired
	int ReprojectCloudToImage(xyzCloudPtr cloud_in, Mat& dimage_out, const KinectCamParams& cam = KinectCamParams());


	// Maps a depth image to the colour image space by projecting to a point cloud, applying the transformation then reprojecting to a sparse image
	// [in]  Raw depth image
	// [in]  The Kinect sensor including instrinsic camera parameters and stereo calibration data
	// [out] A sparse depth image at the native resolution of the colour camera. Unknown pixels are 0
	// [ret] The number of points successfully mapped to the image
	int MapKinectv2DepthToColour(Mat& dimage_in, Mat& dimage_out, const KinectCamParams& cam = KinectCamParams());


	// Maps the colour image into the depth camera space through geometric relationships rathern than 3D transform
	// [in]  Original colour image (greyscale or coloured)
	// [in]  Original, synchronised depth image
	// [in]  The Kinect sensor including instrinsic camera parameters and stereo calibration data
	// [out] A colour image at the native resolution of the colour camera transformed to depth frame. Unknown pixels are 0
	void FastMapColourToDepth(Mat& cimage_in, Mat& dimage_in, Mat& cimage_mapped_out, const KinectCamParams& cam = KinectCamParams());


	// Returns the depth of a pixel in the colour image from a depth image map mapped to the colour space
	// If the depth is not available at that image point it is interpolated from depth results in a surrounding
	// 7x7 window. If there are no valid depth results in that window, 0 is returned. 
	// [in]  The depth image mapped to the colour space
	// [in]  Pixel coordinates to find the depth of
	// [ret] Depth of the pixel 
	unsigned short FindDepthInColourImage(Mat& mapped_depth_in, int row, int col);


	// Creates a coloured 3D point in the cloud for every non-zero depth reading in the depth image
	// [in]  The depth image mapped to the colour image space
	// [in]  The number of valid depth points in that image
	// [in]	 The corresponding colour image
	// [in]  Parameters of the colour camera
	// [out] The cloud to output to
	// Requires the depth image to have already been mapped to the colour image space. 
	void ProjectMappingToColouredCloud(Mat& depthImage_in, int num_valid_depth, Mat& colourImage_in, float fx, float fy, float cx, float cy, xyzrgbCloudPtr cloud_out);


	// As above but overloaded to take the default kinect instead
	// Assumes we are mapping depth and colour images both in colour space to a cloud
	void ProjectMappingToColouredCloud(Mat& depthImage_in, int num_valid_depth, Mat& colourImage_in, xyzrgbCloudPtr cloud_out, const KinectCamParams& cam = KinectCamParams());


	// Combines two point clouds into one and colours them red and blue for viewing
	// [in]  The first point cloud
	// [in]  The second point cloud
	// [out] The combined coloured pont cloud
	void CombineAndColourTwoClouds(xyzCloudPtr cloud1_in, xyzCloudPtr cloud2_in, xyzrgbCloudPtr cloud_out);


}

#endif