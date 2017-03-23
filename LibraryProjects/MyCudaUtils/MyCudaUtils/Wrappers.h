#ifndef WRAPPERS_H
#define WRAPPERS_H

void FindMinimisingTransform(float* H_A, float* H_B, int cols, float* stateVec_out);

void DrawClouds(float* cloud_data, int num_pts, float* cloud2_data = NULL, float min_range = 0.5, float max_range = 4.5, char* fileName = NULL);

void DrawNormals(float* cloud_data, int num_pts, float* normal_data, float min = 0.5, float max = 4.5);

void QueryPoint(cv::Mat& correp_mask);

void DrawCorrep(float* trans_cloud_old_data, float* cloud_new_data, int num_pts, int* correp_mask);

void DrawDepthMaps(cv::Mat& dimage_new, cv::Mat& dimage_old);

#endif