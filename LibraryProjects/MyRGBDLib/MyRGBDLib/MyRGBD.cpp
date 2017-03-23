#include "MyRGBD.h"

// Constructor takes input options of what to print / draw for debugging
// [in]  Whether incremental output should be printed to the terminal
// [in]  Whether timing information should be printed to the terminal
// [in]  Whether images of the keypoints detected should be generated and displayed
// [in]  Whether images of the keypoint matches should be generated and displayed
// [in]  Whether the 3D keypoint point clouds should be displayed overlaid
// [in]  Whether the full dense point clouds should be displayed overlaid
MyRGBD::MyRGBD(bool printOutput_in, bool showTiming_in, bool drawKeypoints_in, bool drawMatches_in, bool drawClouds_in, bool drawDenseClouds_in) {
	timing = showTiming_in;
	displayKeypoints = drawKeypoints_in;
	displayMatches = drawMatches_in;
	displayClouds = drawClouds_in;
	displayDenseClouds = drawDenseClouds_in;
	printOutput = printOutput_in;
	saveOutput = false;

	cimage_buf.resize(IMG_BUF_SIZE);	
	dimage_buf.resize(IMG_BUF_SIZE);	
	gimage_buf.resize(IMG_BUF_SIZE);	
	mimage_buf.resize(IMG_BUF_SIZE);

	relative_VO_pose = Eigen::Matrix4f::Identity();
	concat_VO_pose = Eigen::Matrix4f::Identity();

	cimage_cnt = 0;
	dimage_cnt = 0;
	mimage_cnt = 0;
	gimage_cnt = 0;

	//featureDetector = cv::StarFeatureDetector(45, 40, 10, 9, 10);
}



// Initialise the star tracker settings. The default values for each entry are the OpenCV default settings
// If this function is not called, the default values indicated will be used automatically
// [in]  maxSize size of the largest octagonal bi-level filter to apply to the image. Enlarge to detect bigger features
// [in]  responseThresh minimum threshold response required to be considered a keypoint. Enlarge to choose only stronger features
// [in]  lineThreshProjected
// [in]  lineThreshBinarised This and the above eliminate keypoints based upon a strong line response
// [in]  Size of the window in which non-maximum responses will be suppressed. Enlarge to reduce proximity of features
void MyRGBD::InitStarTracker(int maxSize, int responseThresh, int lineThreshProjected, int lineThresBinarised, int suppressNonmaxSize) {
	featureDetector = cv::StarFeatureDetector(maxSize, responseThresh, lineThreshProjected, lineThresBinarised, suppressNonmaxSize);
}



// Initalises output to the given filename
// No checking is performed that the path is valid
// If this function is not called, VO poses will not output to any file but can still output to console
// [in]  
// [in]  Full name of file to write to including path
// [ret] Success of opening the given files
bool MyRGBD::InitOutput(char* posesFileName, char* timestampFileName){
	poseFile.open(posesFileName);
	timestampsFile.open(timestampFileName);
	if (poseFile.is_open() && timestampsFile.is_open()) {
		saveOutput = true;
		return true;
	}
	else
		return false;
}



// Returns references to the heads of the internal buffer 
// Use these functions to pass in data to the buffers  
// [ret] A reference to the next image in the buffer 
//       This is an empty image on first time loading the buffer, oldest image otherwise
cv::Mat& MyRGBD::NextCImage(void) {
	cv::Mat& nextCimage = cimage_buf[(cimage_cnt++) % IMG_BUF_SIZE];
	return nextCimage;
}
cv::Mat& MyRGBD::NextDImage(void) {
	cv::Mat& nextDimage = dimage_buf[(dimage_cnt++) % IMG_BUF_SIZE];
	return nextDimage;
}



// Operates on the most recent colour and depth images stored in the buffer
// Requires the buffer to be loaded with the NextCImage / DImage funcs above
// Output pose calculated is stored in the public variables below
void MyRGBD::ProcessColourAndDepth(long long time) {

	// Prepare images for processing
	timer.tic(timing);
	PrepareImages();
	timer.toc("Converting colour and remapping depth image", timing);

	/// Detect and describe features
	timer.tic();
	featureDetector.detect(gimage_new, new_kp);
	featureExtractor.compute(gimage_new, new_kp, new_desc);
	timer.toc("Detecting STAR keypoints and computing BRIEF descriptors", timing);

	// Draw Keypoints 
	if (displayKeypoints) {
		cv::drawKeypoints(gimage_new, new_kp, keypoint_image);
		imshow("Keypoints found", keypoint_image);
		while (cv::waitKey(1) != 'q');
		cv::destroyAllWindows();
	}

	// Break if we don't have enough images yet
	if (!(gimage_cnt > 1 && mimage_cnt > 1))
		return;

	// Filter matches with Lowe's algorithm and near frustum edges
	timer.tic(timing);
	match_idx.create(new_desc.rows, 2, CV_32SC1);
	match_dist.create(new_desc.rows, 2, CV_32FC1);

	BruteHammingSSE(new_desc, old_desc, match_idx, match_dist);

	good_matches.clear();
	for (int i = 0; i < match_dist.rows; i++) {

		if (match_dist.at<float>(i, 0) < 0.6 * match_dist.at<float>(i, 1)) {

			if (new_kp[i].pt.x > 110 && new_kp[i].pt.x < 1884) {
				cv::DMatch dm(i, match_idx.at<int>(i, 0), match_dist.at<float>(i, 0));
				good_matches.push_back(dm);
			}
		}
	}
	timer.toc("Good matches brute-force selected", timing);
	
	// Draw the matches
	if (displayMatches) {
		cv::drawMatches(gimage_new, new_kp, gimage_old, old_kp, good_matches, matched_image);
		imshow("Keypoint matches", matched_image);
		while (cv::waitKey(1) != 'q');
		cv::destroyAllWindows();
	}

	// Project the keypoints to a cloud using the mapped depth image
	timer.tic(timing);
	int validpts = ProjectKeypointsToCloud();
	timer.toc("Keypoints projected to 3D", timing);
	if (printOutput) std::cout << "Number of good matches projected to 3D: " << validpts << std::endl;

	// Align the keypoint clouds
	timer.tic(timing);
	relative_VO_pose = pcl::umeyama(keypoint_cloud_old, keypoint_cloud_new, false);
	//Eigen::Matrix<float, 4, 4> transform_rev = pcl::umeyama(keypoint_cloud_new, keypoint_cloud_old, false);
	//Eigen::Matrix<float, 4, 4> transform_rev_inv = transform_rev.inverse();
	timer.toc("Aligned visual features in 3D using Umeyama point-to-point minimisation", timing);

	// Print results to terminal and files
	if (printOutput){
		concat_VO_pose = concat_VO_pose*relative_VO_pose;
		std::cout << "For image timestamp: " << time << " VO Result: " << std::endl;
		std::cout << relative_VO_pose << std::endl;
		std::cout << "Concatenated pose result" << std::endl;
		std::cout << concat_VO_pose << std::endl;
	}
	if (saveOutput) {
		poseFile << relative_VO_pose << std::endl;
		timestampsFile << time << std::endl;
	}
	

	// Draw clouds for debugging
	if (displayClouds)
		DisplayFeatureClouds();

	if (displayDenseClouds)
		DisplayDenseClouds();
}




void MyRGBD::PrepareImages(void) {
	// Copy keypoint descriptors over
	// TODO: implement a circular buffer for keypoints like the images
	
	old_kp = new_kp;
	old_desc = new_desc;
	
	// Map depth image to colour space
	dimage_new = dimage_buf[(dimage_cnt - 1) % IMG_BUF_SIZE];
	MyKinectv2Utils::MapKinectv2DepthToColour(dimage_new, mimage_buf[(mimage_cnt++) % IMG_BUF_SIZE]);
	mimage_new = mimage_buf[(mimage_cnt - 1) % IMG_BUF_SIZE];

	// Convert colour image to greyscale
	cimage_new = cimage_buf[(cimage_cnt - 1) % IMG_BUF_SIZE];
	if (cimage_new.channels() > 1)
		cv::cvtColor(cimage_new, gimage_buf[(gimage_cnt++) % IMG_BUF_SIZE], CV_BGR2GRAY);
	else
		gimage_buf[(gimage_cnt++) % IMG_BUF_SIZE] = cimage_new;
	gimage_new = gimage_buf[(gimage_cnt - 1) % IMG_BUF_SIZE];


	// Check out for -ve indices to our vector
	if (dimage_cnt > 1) {
		dimage_old = dimage_buf[(dimage_cnt - 2) % IMG_BUF_SIZE];
		mimage_old = mimage_buf[(mimage_cnt - 2) % IMG_BUF_SIZE];
	}
	if (cimage_cnt > 1) {
		cimage_old = cimage_buf[(cimage_cnt - 2) % IMG_BUF_SIZE];
		gimage_old = gimage_buf[(gimage_cnt - 2) % IMG_BUF_SIZE];
	}
	
}

int MyRGBD::ProjectKeypointsToCloud(void) {
	
	int npts = good_matches.size();
	int validpts = 0;
	keypoint_cloud_old.resize(3, npts);
	keypoint_cloud_new.resize(3, npts);


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
		}
	}
	//std::cout << "Valid depth points found: " << validpts << std::endl;

	keypoint_cloud_old.conservativeResize(3, validpts);
	keypoint_cloud_new.conservativeResize(3, validpts);
	return validpts;
}


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

void MyRGBD::EigenToPcl(Eigen::Matrix<float, 3, Eigen::Dynamic>* eigen_in, pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_out) {
	int points = eigen_in->cols();
	pcl_out->resize(points);
	for (int i = 0; i < points; i++) {
		pcl::PointXYZ& pt = pcl_out->points[i];
		pt.x = (*eigen_in)(0, i);
		pt.y = (*eigen_in)(1, i);
		pt.z = (*eigen_in)(2, i);
	}
}

void MyRGBD::DisplayFeatureClouds(void) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_new(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_old(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_new_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colour_combined(new pcl::PointCloud<pcl::PointXYZRGB>);

	EigenToPcl(&keypoint_cloud_old, pcl_cloud_old);
	EigenToPcl(&keypoint_cloud_new, pcl_cloud_new);

	MyKinectv2Utils::CombineAndColourTwoClouds(pcl_cloud_new, pcl_cloud_old, cloud_colour_combined);
	pcl::visualization::CloudViewer viewer("Clouds before alignment");
	viewer.showCloud(cloud_colour_combined);
	while (!viewer.wasStopped()){}

	MyKinectv2Utils::TransformCloud(pcl_cloud_new, pcl_cloud_new_trans, relative_VO_pose.inverse());
	MyKinectv2Utils::CombineAndColourTwoClouds(pcl_cloud_new_trans, pcl_cloud_old, cloud_colour_combined);
	pcl::visualization::CloudViewer viewer2("Clouds after alignment");
	viewer2.showCloud(cloud_colour_combined);
	while (!viewer2.wasStopped()){}
}

void MyRGBD::DisplayDenseClouds(void) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr dense_cloud_old(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr dense_cloud_new(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr dense_combined(new pcl::PointCloud<pcl::PointXYZRGB>);
	MyKinectv2Utils::ProjectDepthImageToCloud(dimage_new, dense_cloud_new, cam);
	MyKinectv2Utils::ProjectDepthImageToCloud(dimage_old, dense_cloud_old, cam);
	MyKinectv2Utils::CombineAndColourTwoClouds(dense_cloud_old, dense_cloud_new, dense_combined);
	pcl::visualization::CloudViewer viewer3("Clouds before alignment");
	viewer3.showCloud(dense_combined);
	while (!viewer3.wasStopped()){}
}

void MyRGBD::ProcessColourAndDepthSBA(long long time) {

	// Note notation of using frame 1 and 2 should actually be frame 0 and 1

	// Total number of measurements int num_meas = 0;
	// Find keypoints in frame 1 by filling kps[1].keypoints
	// Find keypoints in frame 2 by filling kps[2].keypoints
	// Compute descriptors by filling kp_desc[1]
	// Compute descriptors by filling kp_desc[2]
	// Match descriptors to populate 'N' good_matches
	// Possibly use RANSAC here to prune good matches, where m is index of match k in frame 1
	// Iterate through good matches (k = 1:N)
	// Project the frame 1 keypoints to 3D, kps[1].keypoints[m] 
	// Create new world point world_pt(numframes)
	// Save the projection to this point x,y,z
	// Set the mask world_pt.frames_vis[1 & 2] = 1;
	// Add the measuremens world_pt.cols_vis[i / i-1] = col(i)/col(i-1) respectively, and for row
	// Push the point back to the list  world_pts.push_back(world_pt)
	// Don't need to increment number of points - stored in size of world_pts
	// Increment number of measurements by num_meas += 2

	// For the next i = 3:numframes images:
	// Find keypoints in current frame by filling kps[i].keypoints
	// Compute descriptors by filling kp_desc[i]
	// Match descriptors between kp_desc[i] and kp_desc[i-1] to get the set
	// of good matches
	// Init world ids of current frame kps.init_idxs()
	// Iterate through good matches k = 1:N, where m is index of previous keypoint in match
	// and n is the index of the matching keypoint in the current frame
	// if (idx = kps[i-1].world_idxs[m]) != --1
	// Then copy the index across to the kps[i].world_idxs[n] = idx
	// Add the mask: world_pts[idx].frames_vis[i] = 1;
	// Add the measurements world_pts[idx].col_meas[i] = col etc. for row
	// Increment number of measurements num_meas++;
	// Else we have a new point 
	// Create new world_pt(numframes)
	// Project to coordinates of [i-1] keypoint to 3D using camera intrinsics
	// Rotate to global frame using pose of frame [i-1]
	// Store in new world_pt
	// Add mask world_pt.frames_vis[i/i-1] = 1
	// Add measurements world_pt.cols_vis[2/1] = col(2)/col(1) etc for row
	// idx = world_pts.push_back(world_pt) (idx = world_pts.size())
	// Add the index to the frame keypoints: kps[i].world_idxs[n] = idx
	// And kps[i-1].world_idxs[m]  = idx;
	// Increment num_meas+= 2;

	// At the end combine all the data
	// Allocate motstruct = new float(numframes*cnp + world_pts.size()*pnp)
	// Allocate mask = new float(numframes * world_pts.size())
	// Allocate measurements = new float(2 * num
	// Concatenate all the poses in to motstruct converting rotationMats to qVecs
	// Iterate through all world_pts
	// Concatenate all the 3D pts into motstruct
	// Concatenate all the masks into vmask
	// Concatenate all the valid measurements (where mask = 1) in order
	// Create the covarience matrix
	// Make sure all the numbers are stored as well
	// Call sba finally
	// Get the output poses and transform back to rotationMats
	// Output the adjusted poses

	// Note we could have a function for adding a new point to stop duplication, but meh
}