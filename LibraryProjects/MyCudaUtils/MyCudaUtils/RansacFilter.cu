#include "MyCudaUtils.h"

using std::vector;

float* D_cloud, *D_kps_new, *D_trans, *D_camPams;
double *D_distArray;
int num_pts;
//__global__ void RansacKernel() {}

__device__ int inlier_count;

__device__ double min_dist_sqd;

__global__ void RansacReprojDistKernel(float* cloud_old, float* kps_new, float* T, double* dist_array, float* camPams, int num_pts) {

	int pt_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (pt_idx >= num_pts)
		return;

	float pt[3] = { 0.0 };
	pt[0] = cloud_old[3 * pt_idx + 0];
	pt[1] = cloud_old[3 * pt_idx + 1];
	pt[2] = cloud_old[3 * pt_idx + 2];

	float predicted_pt[3] = { 0.0 };

	predicted_pt[0] = T[0] * pt[0] + T[4] * pt[1] + T[8] * pt[2] + T[12];	// R00*xi + R01*yi + R02*zi + t_x  
	predicted_pt[1] = T[1] * pt[0] + T[5] * pt[1] + T[9] * pt[2] + T[13];	// R10*xi + R11*yi + R12*zi + t_y  
	predicted_pt[2] = T[2] * pt[0] + T[6] * pt[1] + T[10] * pt[2] + T[14];	// R20*xi + R21*yi + R22*zi + t_z 

	float invZ = 1.0 / predicted_pt[2];
	float pred_col = (predicted_pt[0] * camPams[CFX] * invZ) + camPams[CCX];
	float pred_row = (predicted_pt[1] * camPams[CFY] * invZ) + camPams[CCY];

	float meas_row = kps_new[pt_idx];
	float meas_col = kps_new[pt_idx + num_pts];

	double dist = (meas_row - pred_row)*(meas_row - pred_row) + (meas_col - pred_col)*(meas_col - pred_col);

	dist_array[pt_idx] = dist;
	if (dist < min_dist_sqd) {
		atomicAdd(&inlier_count, 1);
	}
	
}

cudaError_t MyCudaUtils::RansacLoadCloud(float* cloud_old, float* kps_new, int num_pts_in, float* camPams, double min_dist_sqd_in) {
	
	num_pts = num_pts_in;
	
	try {
		CUDA_CHECK(cudaSetDevice(0));
		
		CUDA_INIT_MEM(D_camPams, camPams, CAMPAMSZ * sizeof(float));

		CUDA_INIT_MEM(D_cloud, cloud_old, num_pts * 3 * sizeof(float));

		CUDA_INIT_MEM(D_kps_new, kps_new, num_pts * 2 * sizeof(float));

		CUDA_MALLOC(D_distArray, num_pts * sizeof(double));

		CUDA_MALLOC(D_trans, 4 * 4 * sizeof(float));

		CUDA_CHECK(cudaMemcpyToSymbol(min_dist_sqd, &min_dist_sqd_in, sizeof(double)));
	}
	catch (cudaError_t cudaStatus) {
		cudaFree(D_cloud);
		cudaFree(D_kps_new);
		cudaFree(D_camPams);
		cudaFree(D_distArray);
		cudaFree(D_trans);
		return cudaStatus;
	}
	return cudaSuccess;
}

cudaError_t MyCudaUtils::RansacCalcReprojDist(float* transform) {
	try {
		
		CUDA_UPLOAD(D_trans, transform, 4 * 4 * sizeof(float));

		int inliers = 0;

		CUDA_CHECK(cudaMemcpyToSymbol(inlier_count, &inliers, sizeof(int)));

		// Launch the Kernel
		int blocks = cvCeil(num_pts / 128.0f);
		RansacReprojDistKernel<<<blocks, 128>>>(D_cloud, D_kps_new, D_trans, D_distArray, D_camPams, num_pts);

	}
	catch (cudaError_t cudaStatus) {
		cudaFree(D_trans);
		cudaFree(D_cloud);
		cudaFree(D_kps_new);
		cudaFree(D_camPams);
		cudaFree(D_distArray);

		return cudaStatus;
	}
	return cudaSuccess;
}

cudaError_t MyCudaUtils::RansacGetInlierCount(int* inliers, double* distArray) {
	try {
		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpyFromSymbol(inliers, inlier_count, sizeof(int)));

		if (distArray != NULL) {
			CUDA_DOWNLOAD(distArray, D_distArray, num_pts * sizeof(double));
		}

	}
	catch (cudaError_t cudaStatus){
		cudaFree(D_trans);
		cudaFree(D_cloud);
		cudaFree(D_camPams);
		cudaFree(D_kps_new);
		cudaFree(D_distArray);

		return cudaStatus;
	}

	return cudaSuccess;
}

cudaError_t MyCudaUtils::RansacGetDists(double* distArray) {
	try {
		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());
		
		CUDA_DOWNLOAD(distArray, D_distArray, num_pts * sizeof(double));

	}
	catch (cudaError_t cudaStatus){
		cudaFree(D_trans);
		cudaFree(D_cloud);
		cudaFree(D_camPams);
		cudaFree(D_kps_new);
		cudaFree(D_distArray);

		return cudaStatus;
	}
}

void MyCudaUtils::RansacTidy(void) {

	cudaFree(D_trans);
	cudaFree(D_cloud);
	cudaFree(D_camPams);
	cudaFree(D_kps_new);
	cudaFree(D_distArray);
}