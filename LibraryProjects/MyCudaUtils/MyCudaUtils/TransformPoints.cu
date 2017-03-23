

#include "MyCudaUtils.h"


__global__ void TransformKernel(float* in, float* T, float* out, int num_pts) {
	
	int pt = blockIdx.x * blockDim.x + threadIdx.x;
	if (pt < num_pts) {
		out[3 * pt + 0] = T[0] * in[3 * pt + 0] + T[4] * in[3 * pt + 1] + T[8] * in[3 * pt + 2] + T[12];	// R00*xi + R01*yi + R02*zi + t_x  
		out[3 * pt + 1] = T[1] * in[3 * pt + 0] + T[5] * in[3 * pt + 1] + T[9] * in[3 * pt + 2] + T[13];	// R10*xi + R11*yi + R12*zi + t_y  
		out[3 * pt + 2] = T[2] * in[3 * pt + 0] + T[6] * in[3 * pt + 1] + T[10] * in[3 * pt + 2] + T[14];	// R20*xi + R21*yi + R22*zi + t_z 
	}
}


cudaError_t MyCudaUtils::TransformPoints(float* points_in, float* transform, float* points_out, int num_pts) {

	float* D_pts_in;
	float* D_pts_out;
	float* D_transform;

	int arraySzBytes = num_pts * 3 * sizeof(float);

	int transSzBytes = 4 * 4 * sizeof(float);
	
	try {

		CUDA_CHECK(cudaSetDevice(0));

		CUDA_CHECK(cudaMalloc((void**)&D_pts_in, arraySzBytes));

		CUDA_CHECK(cudaMalloc((void**)&D_pts_out, arraySzBytes));

		CUDA_CHECK(cudaMalloc((void**)&D_transform, 4 * 4 * sizeof(float)));

		CUDA_CHECK(cudaMemcpy(D_pts_in, points_in, arraySzBytes, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(D_transform, transform, transSzBytes, cudaMemcpyHostToDevice));

		int blocks = cvCeil(num_pts / 128);
		TransformKernel << < blocks, 128 >> >(D_pts_in, D_transform, D_pts_out, num_pts);

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(points_out, D_pts_out, arraySzBytes, cudaMemcpyDeviceToHost));

		throw (cudaSuccess);
	}
	catch (cudaError_t cudaStatus) {

		cudaFree(D_pts_in);
		cudaFree(D_pts_out);
		cudaFree(D_transform);

		return cudaStatus;

	}

}
