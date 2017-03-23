#include "MyDrawingUtils.h"


void MyDrawingUtils::DrawKeypoints(Mat& gimage_in, vector<KeyPoint>& kps_in) {
	Mat keypoint_image;
	drawKeypoints(gimage_in, kps_in, keypoint_image);
	imshow("Keypoints found", keypoint_image);
	while (waitKey(1) != 'q');
	destroyAllWindows();
}



void MyDrawingUtils::DrawMatches(vector<Mat>& gimages, vector<frame_kps>& kps, vector<DMatch>& good_matches, int current_frame) {
	Mat matched_image;
	drawMatches(gimages[current_frame], kps[current_frame].keypoints, gimages[current_frame - 1], kps[current_frame - 1].keypoints, good_matches, matched_image);
	imshow("Keypoint matches", matched_image);
	while (cv::waitKey(1) != 'q');
	cv::destroyAllWindows();
}



void MyDrawingUtils::DrawKeypointTracks(Mat& gimage_current, std::vector<world_pt>& world_pts, int numframes, int startframe, int increment) {
	const int LINE_AA = 16;
	Mat tmp, tracks_img;
	gimage_current.copyTo(tmp);
	cv::cvtColor(tmp, tracks_img, CV_GRAY2BGR);
	cv::RNG& rng = cv::theRNG();
	for (int i = 0; i < world_pts.size(); i += increment) {
		cv::Scalar colour = cv::Scalar(rng(256), rng(256), rng(256));
		std::vector<cv::Point> kp_locations;
		// Note cv::Point is doubly-deffed from Point_<type>; Point2i is a 2D integer Point_<int> 
		// and cv::Point is deffed as Point2i
		for (int j = startframe; j < numframes; j++) {
			if (world_pts[i].frames_vis[j] == 1) {
				kp_locations.push_back(cv::Point(world_pts[i].col_meas[j], world_pts[i].row_meas[j]));
			}
		}
		for (int j = 1; j < kp_locations.size(); j++) {
			cv::line(tracks_img, kp_locations[j], kp_locations[j - 1], colour, 1, LINE_AA);
		}
	}

	imshow("Keypoint matches", tracks_img);
	while (cv::waitKey(1) != 'q');
	cv::destroyAllWindows();
}


void MyDrawingUtils::EigenToPcl(Eigen::Matrix<float, 3, Eigen::Dynamic>& eigen_in, pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_out) {
	int points = eigen_in.cols();
	pcl_out->resize(points);
	for (int i = 0; i < points; i++) {
		pcl::PointXYZ& pt = pcl_out->points[i];
		pt.x = eigen_in(0, i);
		pt.y = eigen_in(1, i);
		pt.z = eigen_in(2, i);
	}
}

void MyDrawingUtils::DisplayFeatureClouds(vector<Eigen::Matrix<float, 3, Eigen::Dynamic>>& kp_clouds, Eigen::Matrix<float, 4, 4> relative_pose, int current_frame) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_new(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_old(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_new_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colour_combined(new pcl::PointCloud<pcl::PointXYZRGB>);

	EigenToPcl(kp_clouds[current_frame - 1], pcl_cloud_old);
	EigenToPcl(kp_clouds[current_frame], pcl_cloud_new);

	MyKinectv2Utils::CombineAndColourTwoClouds(pcl_cloud_new, pcl_cloud_old, cloud_colour_combined);
	pcl::visualization::CloudViewer viewer("Clouds before alignment");
	viewer.showCloud(cloud_colour_combined);
	while (!viewer.wasStopped()){}

	MyKinectv2Utils::TransformCloud(pcl_cloud_new, pcl_cloud_new_trans, relative_pose.inverse());
	MyKinectv2Utils::CombineAndColourTwoClouds(pcl_cloud_new_trans, pcl_cloud_old, cloud_colour_combined);
	pcl::visualization::CloudViewer viewer2("Clouds after alignment");
	viewer2.showCloud(cloud_colour_combined);
	while (!viewer2.wasStopped()){}
}

void MyDrawingUtils::DisplayDenseClouds(vector<Mat>& dimages, int current_frame) {
	KinectCamParams cam;
	pcl::PointCloud<pcl::PointXYZ>::Ptr dense_cloud_old(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr dense_cloud_new(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr dense_combined(new pcl::PointCloud<pcl::PointXYZRGB>);
	MyKinectv2Utils::ProjectDepthImageToCloud(dimages[current_frame], dense_cloud_new, cam);
	MyKinectv2Utils::ProjectDepthImageToCloud(dimages[current_frame - 1], dense_cloud_old, cam);
	MyKinectv2Utils::CombineAndColourTwoClouds(dense_cloud_old, dense_cloud_new, dense_combined);
	pcl::visualization::CloudViewer viewer3("Clouds before alignment");
	viewer3.showCloud(dense_combined);
	while (!viewer3.wasStopped()){}
}