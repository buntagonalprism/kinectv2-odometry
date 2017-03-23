#include "MyKinectv2Utils.h"


// Project a depth image to a point cloud using a set of camera intrinsics
// [in]  The depth image to project, with depths in mm
// [in]  The intrinsic camera parameters which produced the depth image
// [out] The output cloud, with pionts in m
void MyKinectv2Utils::ProjectDepthImageToCloud(Mat& depthImage_in, xyzCloudPtr cloud_out, float fx, float fy, float cx, float cy) {
	cloud_out->width = depthImage_in.cols;
	cloud_out->height = depthImage_in.rows;
	cloud_out->resize(cloud_out->height * cloud_out->width);

	// Fill in the point clouds by projecting the depth images
	float focal_x = 1.0f / fx;
	float focal_y = 1.0f / fy;
	int pt_id = 0;

	for (int row = 0; row < depthImage_in.rows; row++)
	{
		for (int col = 0; col < depthImage_in.cols; col++, pt_id++)
		{
			pcl::PointXYZ& pt = cloud_out->points[pt_id];
			uint16_t depth = depthImage_in.at<uint16_t>(row, col);
			if (depth > 10000 || depth < 0)
				depth = 0;
			pt.z = depth * 0.001f;
			pt.x = static_cast<float> (col - cx) * pt.z * focal_x;
			pt.y = static_cast<float> (row - cy) * pt.z * focal_y;
		}
	}
}


// As above but overloaded to take the default kinect instead
// Assumes projection from the depth camera
void MyKinectv2Utils::ProjectDepthImageToCloud(Mat& depthImage_in, xyzCloudPtr cloud_out, const KinectCamParams& cam ) {
	ProjectDepthImageToCloud(depthImage_in, cloud_out, cam.depth_fx, cam.depth_fy, cam.depth_cx, cam.depth_cy);
}


// Projects a colour pixel to an Eigen-style point using default colour camera intrinsics
// [in]  The depth image mapped to the colour space
// [in]  The pixel coordinates to project
// [out] The 3D coordinates of the point
// [ret] Success or failure (failure occurs if there is no depth data at the pixel location)
bool MyKinectv2Utils::ProjectColourPixelToPoint(Mat& depth_mapped_in, int row, int col, Eigen::Matrix < float, 3, 1 >& pt_out, const KinectCamParams& cam ) {
	unsigned short depth = FindDepthInColourImage(depth_mapped_in, row, col);
	if (depth == 0)
		return false;
	float focal_x = 1.0f / cam.colour_fx;
	float focal_y = 1.0f / cam.colour_fy;
	pt_out(0) = 0.001* static_cast<float> ((col - cam.colour_cx) * depth) * focal_x;
	pt_out(1) = 0.001* static_cast<float> ((row - cam.colour_cy) * depth) * focal_y;
	pt_out(2) = 0.001* static_cast<float> (depth);
	return true;

}


// Transform a point cloud by a homogenous transform
// [in]  The input cloud
// [in]  The 4x4 homogenous transformation matrix to apply
// [out] The output cloud
void MyKinectv2Utils::TransformCloud(xyzCloudPtr cloud_in, xyzCloudPtr cloud_out, Eigen::Matrix4f transform) {
	pcl::transformPointCloud(*cloud_in, *cloud_out, transform);
}


// As above but overloaded to take the defaul kinect instead
// Assumes transformation from colour to depth camera frames
void MyKinectv2Utils::TransformCloud(xyzCloudPtr cloud_in, xyzCloudPtr cloud_out, const KinectCamParams& cam ){
	TransformCloud(cloud_in, cloud_out, cam.depthToColourTransform);
}


// Projects a cloud back to an image using the specified camera intrinsics and image dimensions
// [in]  Cloud to reproject back to an image
// [in]  Dimensions of the image to form
// [in]  Intrinsic camera parameters of the camera observing the image
// [out] Depth image of the cloud of specified dimensions taken by specified camera
// [ret] The number of depth points successfully mapped to the colour image 
int MyKinectv2Utils::ReprojectCloudToImage(xyzCloudPtr cloud_in, Mat& dimage_out, float fx, float fy, float cx, float cy, int num_rows, int num_cols) {
	dimage_out = Mat::zeros(num_rows, num_cols, CV_16UC1);
	int size = cloud_in->size();
	int row, col;
	int pts_mapped = 0;
	float invZ;
	for (int i = 0; i < size; i++) {
		pcl::PointXYZ& pt = cloud_in->points[i];
		invZ = 1.0 / pt.z;
		col = boost::math::round((pt.x * fx * invZ) + cx);
		row = boost::math::round((pt.y * fy * invZ) + cy);
		if (row < num_rows && row >= 0 && col >= 0 && col < num_cols) {
			dimage_out.at<unsigned short>(row, col) = (unsigned short)(pt.z * 1000);
			pts_mapped++;
		}
	}

	return pts_mapped;
}


// As above but overloaded to take the default kinect instead
// Assumes that reprojection back to the colour image is desired
int MyKinectv2Utils::ReprojectCloudToImage(xyzCloudPtr cloud_in, Mat& dimage_out, const KinectCamParams& cam ) {
	return ReprojectCloudToImage(cloud_in, dimage_out, cam.colour_fx, cam.colour_fy, cam.colour_cx, cam.colour_cy, cam.cColorHeight, cam.cColorWidth);
}


// Maps a depth image to the colour image space by projecting to a point cloud, applying the transformation then reprojecting to a sparse image
// [in]  Raw depth image
// [in]  The Kinect sensor including instrinsic camera parameters and stereo calibration data
// [out] A sparse depth image at the native resolution of the colour camera. Unknown pixels are 0
// [ret] The number of points successfully mapped to the image
int MyKinectv2Utils::MapKinectv2DepthToColour(Mat& dimage_in, Mat& dimage_out, const KinectCamParams& cam) {

	xyzCloudPtr depth_space_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	xyzCloudPtr colour_space_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	ProjectDepthImageToCloud(dimage_in, depth_space_cloud, cam.depth_fx, cam.depth_fy, cam.depth_cx, cam.depth_cy);

	TransformCloud(depth_space_cloud, colour_space_cloud, cam.depthToColourTransform);

	int pts_mapped = ReprojectCloudToImage(colour_space_cloud, dimage_out, cam.colour_fx, cam.colour_fy, cam.colour_cx, cam.colour_cy, cam.cColorHeight, cam.cColorWidth);

	return pts_mapped;

}


// Returns the depth of a pixel in the colour image from a depth image map mapped to the colour space
// If the depth is not available at that image point it is interpolated from depth results in a surrounding
// 7x7 window. If there are no valid depth results in that window, 0 is returned. 
// [in]  The depth image mapped to the colour space
// [in]  Pixel coordinates to find the depth of
// [ret] Depth of the pixel 
unsigned short MyKinectv2Utils::FindDepthInColourImage(Mat& mapped_depth_in, int row, int col) {

	unsigned short pixel_depth = mapped_depth_in.at<unsigned short>(row, col);
	if (pixel_depth > 0)
		return pixel_depth;

	int valid_depths = 0;
	float depth_sum = 0.0;
	float weighting = 0.0;
	float weighting_sum = 0.0;
	for (int i = -5; i <= 5; i++){
		for (int j = -5; j <= 5; j++) {
			unsigned short depth = mapped_depth_in.at<unsigned short>(row + i, col + j);
			if (depth > 0) {
				weighting = (1.0 / (sqrt(i*i + j*j)));
				depth_sum += weighting*depth;
				weighting_sum += weighting;
				valid_depths++;
			}
		}
	}
	if (valid_depths == 0)
		return 0;
	
	pixel_depth = boost::math::round(depth_sum / weighting_sum);
	return pixel_depth;
}


// Creates a coloured 3D point in the cloud for every non-zero depth reading in the depth image
// [in]  The depth image mapped to the colour image space
// [in]  The number of valid depth points in that image
// [in]	 The corresponding colour image
// [in]  Parameters of the colour camera
// [out] The cloud to output to
// Requires the depth image to have already been mapped to the colour image space. 
void MyKinectv2Utils::ProjectMappingToColouredCloud(Mat& depthImage_in, int num_valid_depth, Mat& colourImage_in, float fx, float fy, float cx, float cy, xyzrgbCloudPtr cloud_out) {

	cloud_out->resize(num_valid_depth);

	// Fill in the point clouds by projecting the depth image
	float focal_x = 1.0f / fx;
	float focal_y = 1.0f / fy;
	int pt_id = 0;

	for (int row = 0; row < colourImage_in.rows; row++)
	{
		for (int col = 0; col < colourImage_in.cols; col++)
		{
			// Only project when we have depth data
			uint16_t depth = depthImage_in.at<uint16_t>(row, col);
			if (depth > 0) {
				pcl::PointXYZRGB& pt = cloud_out->points[pt_id++];
				cv::Vec3b bgrPixel = colourImage_in.at<cv::Vec3b>(row, col);		// Note that opencv default is BGR representation
				if (depth > 10000 || depth < 0)
					depth = 0;
				pt.z = depth * 0.001f;
				pt.x = static_cast<float> (col - cx) * pt.z * focal_x;
				pt.y = static_cast<float> (row - cy) * pt.z * focal_y;
				pt.r = bgrPixel[2];
				pt.g = bgrPixel[1];
				pt.b = bgrPixel[0];
			}
		}
	}

}


// As above but overloaded to take the default kinect instead
// Assumes we are mapping depth and colour images both in colour space to a cloud
void MyKinectv2Utils::ProjectMappingToColouredCloud(Mat& depthImage_in, int num_valid_depth, Mat& colourImage_in, xyzrgbCloudPtr cloud_out, const KinectCamParams& cam) {
	ProjectMappingToColouredCloud(depthImage_in, num_valid_depth, colourImage_in, cam.colour_fx, cam.colour_fy, cam.colour_cx, cam.colour_cy, cloud_out);
}



// Combines two point clouds into one and colours them red and blue for viewing
// [in]  The first point cloud
// [in]  The second point cloud
// [out] The combined coloured pont cloud
void MyKinectv2Utils::CombineAndColourTwoClouds(xyzCloudPtr cloud1, xyzCloudPtr cloud2, xyzrgbCloudPtr clouds_combined) {
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



// Maps the colour image into the depth camera space through geometric relationships rathern than 3D transform
// [in]  Original colour image (greyscale or coloured)
// [in]  Original, synchronised depth image
// [in]  The Kinect sensor including instrinsic camera parameters and stereo calibration data
// [out] A colour image at the native resolution of the colour camera transformed to depth frame. Unknown pixels are 0
void MyKinectv2Utils::FastMapColourToDepth(Mat& cimage_in, Mat& dimage_in, Mat& cimage_mapped_out, const KinectCamParams& cam) {
	// Input is colour or not
	bool coloured = (cimage_in.channels() == 3);
	
	// Shortcut variables 
	int drows = dimage_in.rows, dcols = dimage_in.cols;
	int crows = cimage_in.rows, ccols = cimage_in.cols;
	float cx_c = cam.colour_cx, cy_c = cam.colour_cy, fx_c = cam.colour_fx, fy_c = cam.colour_fy;
	float cx_d = cam.depth_cx,  cy_d = cam.depth_cy,  fx_d = cam.depth_fx,  fy_d = cam.depth_fy;
	float fx_d_inv = 1 / fx_d, fy_d_inv = 1 / fy_d, inv_z_c = 0.0;

	// Extrinsic parameters between the cameras
	Eigen::Matrix4f transform = cam.depthToColourTransform;
	float deltaX = transform(0, 3);
	float deltaZ = transform(2, 3);

	// For range readings
	unsigned short d;
	bool valid = false;

	// 3D coordinates 
	float x_d, y_d, z_d, x_c, y_c, z_c;

	// Pixel coordinates (v,u) is (row, column)
	int v_d, u_d, v_c, u_c;

	// Allocate the output image
	if (coloured) 
		cimage_mapped_out.create(drows, dcols, CV_8UC3);
	else 
		cimage_mapped_out.create(drows, dcols, CV_8UC1);
	
	// Loop through the depth image
	for (v_d = 0; v_d < drows; v_d++) {
		for (u_d = 0; u_d < dcols; u_d++) {
				
			// Project the depth pixel to 3D 
			d = dimage_in.at<unsigned short>(v_d, u_d);
			valid = (d != -0);
			if (valid) {
				z_d = 0.001*(float)d;
				x_d = (u_d - cx_d) * fx_d_inv * z_d;
				y_d = (v_d - cy_d) * fy_d_inv * z_d;

				// Apply the offset between them
				x_c = x_d + deltaX;
				z_c = z_d + deltaZ;
				y_c = y_d;				// Negligible y offset

				inv_z_c = 1 / z_c;		// To avoid performing two divisions

				// Project back to image coordinates
				u_c = boost::math::round(cx_c + (fx_c * x_c * inv_z_c));
				v_c = boost::math::round(cy_c + (fy_c * y_c * inv_z_c));

				// Check the point lies in the colour image
				if (u_c > (ccols-1) || u_c < 0 || v_c > (crows-1) || v_c < 0)
					valid = false;
			}
				
			// Get the colour / intensity at that point and insert in output mapped image
			if (valid && coloured) 
				cimage_mapped_out.at<cv::Vec3b>(v_d, u_d) = cimage_in.at<cv::Vec3b>(v_c, u_c);
			else if (valid) 
				cimage_mapped_out.at<unsigned char>(v_d, u_d) = cimage_in.at<unsigned char>(v_c, u_c);
			
			// Otherwise set to zero
			else if (coloured) {
				cv::Vec3b px = cimage_mapped_out.at<cv::Vec3b>(v_d, u_d);
				px[0] = 0; px[1] = 0; px[2] = 0;
			}
			else {
				cimage_mapped_out.at<unsigned char>(v_d, u_d) = 0;
			}
		}
	}
}
