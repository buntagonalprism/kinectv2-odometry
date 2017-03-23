

#ifndef MY_SBA_H
#define MY_SBA_H

#include "MyOdometryDatatypes.hpp"
#include "sba.h"
#include "RotMath.h"
#include "ImgProj.h"

#include <opencv2/features2d/features2d.hpp>

#include <pcl/common/eigen.h>

#include <vector>
using std::vector;

#define NUM_FRAMES 10	// Backup definition
#define MAXITER2   150  // Another definition 

namespace MySBA {

	struct SBAParams {
		globs_ globs;
		int cnp, pnp, mnp;
		int numframes;
		double opts[SBA_OPTSSZ], info[SBA_INFOSZ];
		int verbose;
	};

	void AddPointsAndMeasurements(vector<world_pt>& world_pts, vector<frame_kps>& keypoints, vector<cv::DMatch>& matches,
		vector<Eigen::Matrix<float, 3, Eigen::Dynamic>>& kp_clouds, vector<Eigen::Matrix4f>& cam_poses,
		int& num_meas, int frame, int numframes);

	void BundleAdjust(vector<world_pt>& world_pts, vector<Eigen::Matrix4f>& cam_poses, vector<Eigen::Matrix4f>& ba_poses, int num_meas, SBAParams& sbaParams);

	void WriteRelativePoses(std::ostream& output, vector<Eigen::Matrix4f> ba_poses);
}

#endif