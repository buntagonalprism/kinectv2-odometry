
#include "MyCudaUtils.h"
#include "Wrappers.h"
#include "armadillo.h"
#include "pcl\visualization\cloud_viewer.h"
#include "pcl\visualization\pcl_visualizer.h"
#include <memory>


		// Allocate for the clouds
void FindMinimisingTransform(float* H_A, float* H_B, int cols, float* stateVec_out) {
	arma::Mat<float> Amat(6, 6);
	Amat.fill(0.0);
	arma::Mat<float> Bmat(6, 1);
	Bmat.fill(0.0);
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < 36; j++) {
			float x = Amat(j);
			Amat(j) += H_A[36 * i + j];
			x = Amat(j);
		}
		for (int j = 0; j < 6; j++) {
			Bmat(j) += H_B[6 * i + j];
		}
	}

	// Print out the combined matrices for debugging
	//for (int i = 0; i < 6; i++) {
	//	for (int j = 0; j < 6; j++){
	//		printf("%f, ", Amat(i,j));
	//	}
	//	printf("\n");
	//}
	//printf("\n");


	//for (int i = 0; i < 6; i++) {
	//	printf("%f\n", Bmat(i));
	//}
	// Compute the SVD using arma library to solve for x [alp, bet, gam, tx, ty, tz]
	arma::Mat<float> stateVec;
	stateVec = arma::solve(Amat, Bmat);

	for (int i = 0; i < 6; i++) {
		stateVec_out[i] = stateVec(i);
	}

}


void CombineAndColourTwoClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2, pcl::PointCloud<pcl::PointXYZRGB>::Ptr clouds_combined) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloured1(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloured2(new pcl::PointCloud<pcl::PointXYZRGB>);

	coloured1->width = cloud1->width;
	coloured2->width = cloud2->width;
	coloured1->height = cloud1->height;
	coloured2->height = cloud2->height;
	coloured1->resize(coloured1->width * coloured1->height);
	coloured2->resize(coloured2->width * coloured2->height);

	int pt_id = 0;
	for (int row = 0; row < cloud1->width; row++)
	{
		for (int col = 0; col < cloud1->height; col++)
		{
			pcl::PointXYZ& pt = cloud1->points[pt_id];
			pcl::PointXYZRGB& c_pt = coloured1->points[pt_id++];
			c_pt.z = pt.z;
			c_pt.x = pt.x;
			c_pt.y = pt.y;
			c_pt.b = 255;
			c_pt.g = 0;
			c_pt.r = 0;
		}
	}

	pt_id = 0;
	for (int row = 0; row < cloud2->width; row++)
	{
		for (int col = 0; col < cloud2->height; col++)
		{
			pcl::PointXYZ& pt = cloud2->points[pt_id];
			pcl::PointXYZRGB& c_pt = coloured2->points[pt_id++];
			c_pt.z = pt.z;
			c_pt.x = pt.x;
			c_pt.y = pt.y;
			c_pt.b = 0;
			c_pt.g = 0;
			c_pt.r = 255;
		}
	}

	(*clouds_combined) = (*coloured1) + (*coloured2);
}


void DrawClouds(float* cloud_data, int num_pts, float* cloud2_data) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->resize(num_pts);

	for (int i = 0; i < num_pts; i++) {
		pcl::PointXYZ& pt = cloud->points[i];
		pt.x = cloud_data[3 * i + 0];
		pt.y = cloud_data[3 * i + 1];
		pt.z = cloud_data[3 * i + 2];
	}

	if (cloud2_data == NULL) {
		pcl::visualization::CloudViewer viewer("Clouds before alignment");
		viewer.showCloud(cloud);
		while (!viewer.wasStopped()){}
	}
	else {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
		cloud2->resize(num_pts);
		for (int i = 0; i < num_pts; i++) {
			pcl::PointXYZ& pt = cloud2->points[i];
			pt.x = cloud2_data[3 * i + 0];
			pt.y = cloud2_data[3 * i + 1];
			pt.z = cloud2_data[3 * i + 2];
		}

		pcl::visualization::CloudViewer viewer("Clouds before alignment");

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colour_combined(new pcl::PointCloud<pcl::PointXYZRGB>);

		CombineAndColourTwoClouds(cloud, cloud2, cloud_colour_combined);
		
		viewer.showCloud(cloud_colour_combined);
		while (!viewer.wasStopped()){}
	}

}

void ColourByDepth(pcl::PointXYZRGB& pt, float min_range, float max_range) {
	float mid_range = (min_range + max_range) / 2;
	if(pt.z < mid_range && pt.z > min_range) {
		float factor = (pt.z - min_range) / (mid_range - min_range);
		pt.r = 255 - cvRound(factor*(255.0 - 15.0));
		pt.g = 15 + cvRound(factor*(255.0 - 15.0));
		pt.b = 15;
	}
	else if (pt.z < max_range) {
		float factor = (pt.z - mid_range) / (max_range - mid_range);
		pt.r = 15;
		pt.g = 255 - cvRound(factor*(255.0 - 15.0));
		pt.b = 15 + cvRound(factor*(255.0 - 15.0));
	}
	else {
		pt.r = pt.b = pt.g = 255;
	}
}

void DrawClouds(float* cloud_data, int num_pts, float* cloud2_data, float min_range, float max_range, char* fileName) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->resize(num_pts);

	for (int i = 0; i < num_pts; i++) {
		pcl::PointXYZRGB& pt = cloud->points[i];
		pt.x = cloud_data[3 * i + 0];
		pt.y = cloud_data[3 * i + 1];
		pt.z = cloud_data[3 * i + 2];
		float factor = (pt.z - min_range) / (max_range - min_range);
		pt.b = 12;
		pt.r = 12 + cvRound((255.0 - 12.0) *factor);
		pt.g = 240;
		//ColourByDepth(pt, min_range, max_range);
	}

	if (cloud2_data == NULL) {
		pcl::visualization::CloudViewer viewer("Clouds before alignment");
		viewer.showCloud(cloud);
		while (!viewer.wasStopped()){}
	}
	else {
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud2->resize(num_pts);
		for (int i = 0; i < num_pts; i++) {
			pcl::PointXYZRGB& pt = cloud2->points[i];
			pt.x = cloud2_data[3 * i + 0];
			pt.y = cloud2_data[3 * i + 1];
			pt.z = cloud2_data[3 * i + 2];
			float factor = (pt.z - min_range) / (max_range - min_range);
			pt.b = 250 - cvRound((250.0-12.0)*factor);
			pt.r = 12 + cvRound((255.0 - 12.0) *factor);
			pt.g = 12;
			//ColourByDepth(pt, min_range, max_range);
		}

		

		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colour_combined(new pcl::PointCloud<pcl::PointXYZRGB>);

		(*cloud_colour_combined) = (*cloud2) + (*cloud);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_colour_combined);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud_colour_combined, rgb, "Two Frames");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Two Frames");
		viewer->addCoordinateSystem(0.05);
		viewer->setCameraPosition(0.0, 0.2, -1.8, 0.02, -0.97, -0.1);
		if (fileName != NULL) {
			viewer->spinOnce();
			std::string file(fileName);
			viewer->saveScreenshot(file);
		}
		//viewer->initCameraParameters();
		while (!viewer->wasStopped()) {
			viewer->spinOnce(100);
		}
		viewer->close();

		// Functioning simple cloud-viewer
		//CombineAndColourTwoClouds(cloud, cloud2, cloud_colour_combined); // Just red and blue colouring
		//pcl::visualization::CloudViewer viewer("Clouds before alignment");
		//viewer.showCloud(cloud_colour_combined);
		//while (!viewer.wasStopped()){}
	}

}

void DrawNormals(float* cloud_data, int num_pts, float* normal_data, float min_range, float max_range) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->resize(num_pts);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	normals->resize(num_pts);

	float mid_range = (min_range + max_range ) / 2;

	for (int i = 0; i < num_pts; i++) {
		pcl::PointXYZRGB& pt = cloud->points[i];
		pt.x = cloud_data[3 * i + 0];
		pt.y = cloud_data[3 * i + 1];
		pt.z = cloud_data[3 * i + 2];
		ColourByDepth(pt, min_range, max_range);
		pcl::Normal& norm = normals->points[i];
		norm.normal_x = normal_data[3 * i + 0];
		norm.normal_y = normal_data[3 * i + 1];
		norm.normal_z = normal_data[3 * i + 2];
	}

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 200, 0.05, "normals");
	viewer->addCoordinateSystem(0.05);
	//viewer->initCameraParameters();
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
	}
	viewer->close();
	
}


void DrawCorrep(float* trans_cloud_old_data, float* cloud_new_data, int num_pts, int* correp_mask) {

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud_old(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_new(new pcl::PointCloud<pcl::PointXYZRGB>);
	trans_cloud_old->resize(num_pts);
	int mapped_pts = 0, correp_idx;
	for (int i = 0; i < num_pts; i++) {
		pcl::PointXYZRGB& pt = trans_cloud_old->points[i];
		pt.x = trans_cloud_old_data[3 * i + 0];
		pt.y = trans_cloud_old_data[3 * i + 1];
		pt.z = trans_cloud_old_data[3 * i + 2];

		correp_idx = correp_mask[i];
		if (correp_idx > 0) {
			mapped_pts++;
			pcl::PointXYZRGB pt_new;
			pt.r = pt_new.r = cvRound(255.0*rand());
			pt.g = pt_new.g = cvRound(255.0*rand());
			pt.b = pt_new.b = cvRound(255.0*rand());
			pt_new.x = cloud_new_data[3 * correp_idx + 0];
			pt_new.y = cloud_new_data[3 * correp_idx + 1];
			pt_new.z = cloud_new_data[3 * correp_idx + 2];
			cloud_new->push_back(pt_new);
		}
		else {
			pt.r = pt.b = pt.g = 255;
		}
	}
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_combined(new pcl::PointCloud<pcl::PointXYZRGB>);
	(*cloud_combined) = (*trans_cloud_old) + (*cloud_new);

	pcl::visualization::CloudViewer viewer("Cloud correspondances");
	viewer.showCloud(cloud_combined);
	while (!viewer.wasStopped()){}
}


void QueryPoint(cv::Mat& correp_mask) {
	bool quit = false;
	int row, col;
	char c;
	while (quit == false) {
		std::cout << "Enter row and column to query" << std::endl;
		std::cout << "Row: ";
		std::cin >> row;
		std::cout << "Col: ";
		std::cin >> col;
		std::cout << "Data at that location: " << correp_mask.at<int>(row, col) << std::endl;
		std::cout << "Press q to quit or any other key for another point: ";
		std::cin >> c;
		if (c == 'q')
			quit = true;
	}
}

void DrawDepthMaps(cv::Mat& dimage_new, cv::Mat& dimage_old) {
	cv::Mat scaleup_new, scaleup_old;
	scaleup_new = dimage_new * 10;
	scaleup_old = dimage_old * 10;
	cv::imshow("New image", scaleup_new);
	while (cv::waitKey(1) != 'q');
	cv::imshow("Old image", scaleup_old);
	while (cv::waitKey(1) != 'q');
	cv::destroyAllWindows();
}