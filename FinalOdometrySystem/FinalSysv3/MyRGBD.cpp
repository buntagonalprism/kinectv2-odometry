#include "MyRGBD.h"



// Matches features between two sets of computed keypoints
// [in]  Feature descriptors in previous frame
// [in]  Feature descriptors for current frame
// [out] For each keypoint in the current frame, the index of its match in the previous frame
// [out] For each match, the distance between the descriptors
void MyRGBD::MatchFeatures(cv::Mat& kp_desc_new_in, cv::Mat& kp_desc_old_in, cv::Mat& match_idx_out, cv::Mat& match_dist_out) {

	match_idx_out.create(kp_desc_new_in.rows, 2, CV_32SC1);
	match_dist_out.create(kp_desc_new_in.rows, 2, CV_32FC1);
	Timer timer;
	//timer.tic();
	BruteHammingSSE(kp_desc_new_in, kp_desc_old_in, match_idx_out, match_dist_out);
	//timer.toc("Serial hamming matching");
	
	//timer.tic();
	//MyCudaUtils::BruteHammingCUDA(kp_desc_new_in, kp_desc_old_in, match_idx_out, match_dist_out);
	//timer.toc("CUDA Hamming Matching");

}



// Performs Brute-Force hamming distance matching between a set of descriptors
// Accelerated by SSE4 instruction POPCNT
// An alternative to FLANN-based matching with more efficient computation
// for small (< 20,000) sets due to SSE acceleration
void MyRGBD::BruteHammingSSE(cv::Mat& new_desc, cv::Mat& old_desc, cv::Mat& match_idx, cv::Mat& match_dist) {
	unsigned long long* pNewBitString64 = (unsigned long long*) new_desc.data;

	// Loop through all new descriptors
	for (int i = 0; i < new_desc.rows; i++) {
		unsigned long long* pOldBitString64 = (unsigned long long*) old_desc.data;
		unsigned int min_dist = std::numeric_limits<unsigned int>::max() - 1;
		unsigned int second_dist = std::numeric_limits<unsigned int>::max();
		unsigned int min_idx = 0;
		unsigned int second_idx = 0;
		unsigned int distance;

		// Compare with all previous descriptors
		for (int j = 0; j < old_desc.rows; j++) {
			distance = 0;

			// Compute the Hamming distance
			for (int k = 0; k < 4; k++) {
				distance += __popcnt64(pNewBitString64[k] ^ (*pOldBitString64));
				pOldBitString64++;
			}
			if (distance < min_dist) {
				second_dist = min_dist;
				second_idx = min_idx;
				min_dist = distance;
				min_idx = j;
			}
			else if (distance < second_dist) {
				second_dist = distance;
				second_idx = j;
			}
		}
		pNewBitString64 += 4;
		match_dist.at<float>(i, 0) = min_dist;
		match_dist.at<float>(i, 1) = second_dist;
		match_idx.at<int>(i, 0) = min_idx;
		match_idx.at<int>(i, 1) = second_idx;
	}
	pNewBitString64 = nullptr;

}



// Filters the matches to select only those which meet Lowe's criteria
// That the match distance of the best match should be 0.6*second best
// Additionally filters keypoints close to the Frustum edges
vector<cv::DMatch> MyRGBD::LowesFilterFeatures(cv::Mat& match_idx_in, cv::Mat& match_dist_in, std::vector<cv::KeyPoint>& kps_new,
	std::vector<cv::KeyPoint>& kps_old, cv::Mat& gimage_new, cv::Mat& gimage_old) {

	vector<cv::DMatch> good_matches;

	for (int i = 0; i < match_dist_in.rows; i++) {

		if (match_dist_in.at<float>(i, 0) < 0.8 * match_dist_in.at<float>(i, 1)) {

			if (kps_new[i].pt.x > 110 && kps_new[i].pt.x < 1884) {
				cv::DMatch dm(i, match_idx_in.at<int>(i, 0), match_dist_in.at<float>(i, 0));
				good_matches.push_back(dm);
			}
		}
	}

	return good_matches;
}



vector<cv::DMatch> MyRGBD::ProjectKeypointsToCloud(vector<cv::DMatch>& good_matches, EigenDynamicCloud& keypoint_cloud_new, EigenDynamicCloud& keypoint_cloud_old,
	cv::Mat& mimage_new, cv::Mat& mimage_old, vector<cv::KeyPoint>& new_kp, vector<cv::KeyPoint>& old_kp) {

	int npts = good_matches.size();
	int validpts = 0;
	keypoint_cloud_old.resize(3, npts);
	keypoint_cloud_new.resize(3, npts);
	KinectCamParams cam;
	vector<cv::DMatch> projected_matches;


	for (int i = 0; i < npts; i++) {

		// Get pixel coordinates of the matches
		int row_old, col_old, row_new, col_new, kp_old_id, kp_new_id;
		kp_old_id = good_matches[i].trainIdx;
		kp_new_id = good_matches[i].queryIdx;

		row_old = (int)old_kp[kp_old_id].pt.y;
		col_old = (int)old_kp[kp_old_id].pt.x;
		row_new = (int)new_kp[kp_new_id].pt.y;
		col_new = (int)new_kp[kp_new_id].pt.x;

		// Project the points to 3D. 
		Eigen::Matrix<float, 3, 1> pt_old, pt_new;

		bool newMapped = MyKinectv2Utils::ProjectColourPixelToPoint(mimage_new, row_new, col_new, pt_new, cam);
		bool oldMapped = MyKinectv2Utils::ProjectColourPixelToPoint(mimage_old, row_old, col_old, pt_old, cam);

		// Check the points are valid
		if (newMapped && oldMapped) {
			keypoint_cloud_old(0, validpts) = pt_old(0);
			keypoint_cloud_old(1, validpts) = pt_old(1);
			keypoint_cloud_old(2, validpts) = pt_old(2);

			keypoint_cloud_new(0, validpts) = pt_new(0);
			keypoint_cloud_new(1, validpts) = pt_new(1);
			keypoint_cloud_new(2, validpts) = pt_new(2);
			validpts++;
			projected_matches.push_back(good_matches[i]);
		}

	}

	keypoint_cloud_old.conservativeResize(3, validpts);
	keypoint_cloud_new.conservativeResize(3, validpts);

	return projected_matches;
}

void MyRGBD::EstimateTransform(EigenDynamicCloud& kp_cloud_new, EigenDynamicCloud& kp_cloud_old, Eigen::Matrix4f& trans_out) {

	trans_out = pcl::umeyama(kp_cloud_new, kp_cloud_old, false);

}


vector<cv::DMatch> MyRGBD::RansacFilterFeatures(vector<cv::DMatch>& matches, vector<cv::KeyPoint>& new_kp, vector<cv::KeyPoint>& old_kp,
	EigenDynamicCloud& kp_cloud_new, EigenDynamicCloud& kp_cloud_old, RansacParams& ranPams)
{

	// get number of matches
	int N = matches.size();

	// clear parameter vector
	vector<int> inlier_idxs;

	// initial RANSAC estimate
	for (int k = 0; k < ranPams.ransac_iterations; k++) {

		// Draw random sample set
		vector<int> active = GetRandomSample(N, ranPams.min_sample_size);

		EigenDynamicCloud old_sample, new_sample;
		old_sample.resize(3, 3); new_sample.resize(3, 3);
		for (int i = 0; i < 3; i++) {
			old_sample.col(i) = kp_cloud_old.col(active[i]);
			new_sample.col(i) = kp_cloud_new.col(active[i]);
		}

		// Find transform for sample set
		Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

		EstimateTransform(new_sample, old_sample, transform);

		// Count number of inliers for this transform
		vector<int> inliers_curr = GetInliers(matches, new_kp, kp_cloud_old, transform, ranPams);

		// Save as maximum set if more inliers
		if (inliers_curr.size() > inlier_idxs.size()) {
			inlier_idxs = inliers_curr;
		}	
	}

	// Copy the inlier set back
	vector<cv::DMatch> inlier_matches;
	EigenDynamicCloud inlier_cloud_new, inlier_cloud_old;
	inlier_cloud_new.resize( 3, inlier_idxs.size() );
	inlier_cloud_old.resize( 3, inlier_idxs.size() );

	for (int i = 0; i < inlier_idxs.size(); i++)  {

		inlier_matches.push_back( matches[inlier_idxs[i]] );

		inlier_cloud_new.col(i) = kp_cloud_new.col( inlier_idxs[i] );

		inlier_cloud_old.col(i) = kp_cloud_old.col( inlier_idxs[i] );

	}

	kp_cloud_old = inlier_cloud_old;
	kp_cloud_new = inlier_cloud_new;

	return inlier_matches;
}


vector<int> MyRGBD::GetInliers(vector<cv::DMatch>& matches, vector<cv::KeyPoint>& new_kp, EigenDynamicCloud& old_cloud, Eigen::Matrix4f& transform, RansacParams& ranPams) {

	assert(matches.size() == old_cloud.cols());

	// Compute predicted cloud by applying transform to old cloud
	EigenDynamicCloud predicted_cloud;
	Eigen::Matrix4f invTransf = transform.inverse();
	predicted_cloud.resize(3, old_cloud.cols());
	//Timer timer;
	//timer.tic();
	MyCudaUtils::TransformPoints(old_cloud.data(), invTransf.data(), predicted_cloud.data(), old_cloud.cols());
	/*timer.toc("Cuda Pt transform");
	timer.tic();
	for (int i = 0; i < old_cloud.cols(); i++) {
		Eigen::Vector4f old_pt = Eigen::Vector4f::Ones(), predicted_pt;
		old_pt(0) = old_cloud(0, i); old_pt(1) = old_cloud(1, i); old_pt(2) = old_cloud(2, i);
		predicted_pt = invTransf * old_pt;
		predicted_cloud(0, i) = predicted_pt(0);
		predicted_cloud(1, i) = predicted_pt(1);
		predicted_cloud(2, i) = predicted_pt(2);
	}
	timer.toc("Serial pt transform");*/
	
	// Compute predicted keypoint coordinates by reprojecting back to image
	vector<cv::KeyPoint> predicted_kp;
	KinectCamParams cam;
	float invZ, row, col;
	for (int i = 0; i < predicted_cloud.cols(); i++) {
		invZ = 1.0 / predicted_cloud(2, i);
		col = (predicted_cloud(0, i) * cam.colour_fx * invZ) + cam.colour_cx;
		row = (predicted_cloud(1, i) * cam.colour_fy * invZ) + cam.colour_cy;
		cv::KeyPoint kp(col, row, 1);
		predicted_kp.push_back(kp);
	}

	// Compare predicted to measured keypoints and save inliers if the distance is small
	double inlier_thresh = 1.0;
	double inlier_thresh_sqd = pow(inlier_thresh, 1);
	vector<int> inliers;
	for (int i = 0; i < matches.size(); i++) {
		int new_idx = matches[i].queryIdx;
		double distsqd = pow(predicted_kp[i].pt.x - new_kp[new_idx].pt.x, 2) + pow(predicted_kp[i].pt.y - new_kp[new_idx].pt.y, 2);
		if (distsqd < ranPams.inlier_thresh_sqd) {
			inliers.push_back(i);
		}
	}

	return inliers;
}

// Selects 'num' numbers from a set of indices 0->N-1 (i.e. N items in total)
vector<int> MyRGBD::GetRandomSample(int N, int num) {

	// init sample and totalset
	vector<int> sample;
	vector<int> totalset;

	// create vector containing all indices
	for (int i = 0; i<N; i++)
		totalset.push_back(i);

	// add num indices to current sample
	sample.clear();
	for (int i = 0; i<num; i++) {
		int j = rand() % totalset.size();
		sample.push_back(totalset[j]);
		totalset.erase(totalset.begin() + j);
	}

	// return sample
	return sample;
}












/*******************************************************************************
*							O L D    F U N C T I O N S
*******************************************************************************/
// From when the class used internally-store data buffers 
// Might re-implement with overloads later

// Returns references to the heads of the internal buffer 
// Use these functions to pass in data to the buffers  
// [ret] A reference to the next image in the buffer 
//       This is an empty image on first time loading the buffer, oldest image otherwise
//cv::Mat& MyRGBD::NextCImage(void) {
//	cv::Mat& nextCimage = cimage_buf[(cimage_cnt++) % IMG_BUF_SIZE];
//	return nextCimage;
//}
//cv::Mat& MyRGBD::NextDImage(void) {
//	cv::Mat& nextDimage = dimage_buf[(dimage_cnt++) % IMG_BUF_SIZE];
//	return nextDimage;
//}

// Operates on the most recent colour and depth images stored in the buffer
//// Requires the buffer to be loaded with the NextCImage / DImage funcs above
//// Output pose calculated is stored in the public variables below
//void MyRGBD::ProcessColourAndDepth(long long time) {
//
//	// Prepare images for processing
//	timer.tic(timing);
//	PrepareImages();
//	timer.toc("Converting colour and remapping depth image", timing);
//
//	/// Detect and describe features
//	timer.tic();
//	featureDetector.detect(gimage_new, new_kp);
//	featureExtractor.compute(gimage_new, new_kp, new_desc);
//	timer.toc("Detecting STAR keypoints and computing BRIEF descriptors", timing);
//
//	// Draw Keypoints 
//	if (displayKeypoints) {
//		cv::drawKeypoints(gimage_new, new_kp, keypoint_image);
//		imshow("Keypoints found", keypoint_image);
//		while (cv::waitKey(1) != 'q');
//		cv::destroyAllWindows();
//	}
//
//	// Break if we don't have enough images yet
//	if (!(gimage_cnt > 1 && mimage_cnt > 1))
//		return;
//
//	// Filter matches with Lowe's algorithm and near frustum edges
//	timer.tic(timing);
//	match_idx.create(new_desc.rows, 2, CV_32SC1);
//	match_dist.create(new_desc.rows, 2, CV_32FC1);
//
//	BruteHammingSSE(new_desc, old_desc, match_idx, match_dist);
//
//	good_matches.clear();
//	for (int i = 0; i < match_dist.rows; i++) {
//
//		if (match_dist.at<float>(i, 0) < 0.6 * match_dist.at<float>(i, 1)) {
//
//			if (new_kp[i].pt.x > 110 && new_kp[i].pt.x < 1884) {
//				cv::DMatch dm(i, match_idx.at<int>(i, 0), match_dist.at<float>(i, 0));
//				good_matches.push_back(dm);
//			}
//		}
//	}
//	timer.toc("Good matches brute-force selected", timing);
//
//	// Draw the matches
//	if (displayMatches) {
//		cv::drawMatches(gimage_new, new_kp, gimage_old, old_kp, good_matches, matched_image);
//		imshow("Keypoint matches", matched_image);
//		while (cv::waitKey(1) != 'q');
//		cv::destroyAllWindows();
//	}
//
//	// Project the keypoints to a cloud using the mapped depth image
//	timer.tic(timing);
//	int validpts = ProjectKeypointsToCloud();
//	timer.toc("Keypoints projected to 3D", timing);
//	if (printOutput) std::cout << "Number of good matches projected to 3D: " << validpts << std::endl;
//
//	// Align the keypoint clouds
//	timer.tic(timing);
//	relative_VO_pose = pcl::umeyama(keypoint_cloud_old, keypoint_cloud_new, false);
//	//Eigen::Matrix<float, 4, 4> transform_rev = pcl::umeyama(keypoint_cloud_new, keypoint_cloud_old, false);
//	//Eigen::Matrix<float, 4, 4> transform_rev_inv = transform_rev.inverse();
//	timer.toc("Aligned visual features in 3D using Umeyama point-to-point minimisation", timing);
//
//	// Print results to terminal and files
//	if (printOutput){
//		concat_VO_pose = concat_VO_pose*relative_VO_pose;
//		std::cout << "For image timestamp: " << time << " VO Result: " << std::endl;
//		std::cout << relative_VO_pose << std::endl;
//		std::cout << "Concatenated pose result" << std::endl;
//		std::cout << concat_VO_pose << std::endl;
//	}
//	if (saveOutput) {
//		poseFile << relative_VO_pose << std::endl;
//		timestampsFile << time << std::endl;
//	}
//
//
//	// Draw clouds for debugging
//	if (displayClouds)
//		DisplayFeatureClouds();
//
//	if (displayDenseClouds)
//		DisplayDenseClouds();
//}
//
//
//void MyRGBD::PrepareImages(void) {
//	// Copy keypoint descriptors over
//	// TODO: implement a circular buffer for keypoints like the images
//
//	old_kp = new_kp;
//	old_desc = new_desc;
//
//	// Map depth image to colour space
//	dimage_new = dimage_buf[(dimage_cnt - 1) % IMG_BUF_SIZE];
//	MyKinectv2Utils::MapKinectv2DepthToColour(dimage_new, mimage_buf[(mimage_cnt++) % IMG_BUF_SIZE]);
//	mimage_new = mimage_buf[(mimage_cnt - 1) % IMG_BUF_SIZE];
//
//	// Convert colour image to greyscale
//	cimage_new = cimage_buf[(cimage_cnt - 1) % IMG_BUF_SIZE];
//	if (cimage_new.channels() > 1)
//		cv::cvtColor(cimage_new, gimage_buf[(gimage_cnt++) % IMG_BUF_SIZE], CV_BGR2GRAY);
//	else
//		gimage_buf[(gimage_cnt++) % IMG_BUF_SIZE] = cimage_new;
//	gimage_new = gimage_buf[(gimage_cnt - 1) % IMG_BUF_SIZE];
//
//
//	// Check out for -ve indices to our vector
//	if (dimage_cnt > 1) {
//		dimage_old = dimage_buf[(dimage_cnt - 2) % IMG_BUF_SIZE];
//		mimage_old = mimage_buf[(mimage_cnt - 2) % IMG_BUF_SIZE];
//	}
//	if (cimage_cnt > 1) {
//		cimage_old = cimage_buf[(cimage_cnt - 2) % IMG_BUF_SIZE];
//		gimage_old = gimage_buf[(gimage_cnt - 2) % IMG_BUF_SIZE];
//	}
//
//}
// int MyRGBD::ProjectKeypointsToCloud(void) {
//	
//	int npts = good_matches.size();
//	int validpts = 0;
//	keypoint_cloud_old.resize(3, npts);
//	keypoint_cloud_new.resize(3, npts);
//
//
//	for (int i = 0; i < npts; i++) {
//
//		// Get pixel coordinates of the matches
//		int row_old, col_old, row_new, col_new, kp_old_id, kp_new_id;
//		kp_old_id = good_matches[i].trainIdx;
//		kp_new_id = good_matches[i].queryIdx;
//
//		row_old = (int)old_kp[kp_old_id].pt.y;
//		col_old = (int)old_kp[kp_old_id].pt.x;
//		row_new = (int)new_kp[kp_new_id].pt.y;
//		col_new = (int)new_kp[kp_new_id].pt.x;
//
//		// Project the points to 3D. 
//		Eigen::Matrix<float, 3, 1> pt_old, pt_new;
//
//		bool newMapped = MyKinectv2Utils::ProjectColourPixelToPoint(mimage_new, row_new, col_new, pt_new, cam);
//		bool oldMapped = MyKinectv2Utils::ProjectColourPixelToPoint(mimage_old, row_old, col_old, pt_old, cam);
//
//		// Check the points are valid
//		if (newMapped && oldMapped) {
//			keypoint_cloud_old(0, validpts) = pt_old(0);
//			keypoint_cloud_old(1, validpts) = pt_old(1);
//			keypoint_cloud_old(2, validpts) = pt_old(2);
//
//			keypoint_cloud_new(0, validpts) = pt_new(0);
//			keypoint_cloud_new(1, validpts) = pt_new(1);
//			keypoint_cloud_new(2, validpts) = pt_new(2);
//			validpts++;
//		}
//	}
//	//std::cout << "Valid depth points found: " << validpts << std::endl;
//
//	keypoint_cloud_old.conservativeResize(3, validpts);
//	keypoint_cloud_new.conservativeResize(3, validpts);
//	return validpts;
//}