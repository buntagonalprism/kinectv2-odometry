#include "MyCudaUtils.h"

#define ULLTO256BIT 4	// How many unsigned long longs in a 256 bit descriptors
#define MATCHSZ 4		// Number of values to describe matches
#define BYTESIN256BIT 32 // Number of bytes in a 256 bit descriptor
#define MAXDIST	256			// Maximum hamming distance in 256 bit descriptor

__global__ void HammingKernel(unsigned long long* new_desc, unsigned long long* old_desc, unsigned int* matches_idx, float* matches_dist, int num_new_pts, int num_old_pts) {
	
	int new_pt_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (!(new_pt_idx < num_new_pts))
		return;
	
	unsigned long long* pNewBitString64 = new_desc + new_pt_idx * ULLTO256BIT;
	unsigned long long* pOldBitString64 = old_desc;

	unsigned int min_dist = MAXDIST + 1;
	unsigned int second_dist = MAXDIST + 2;
	unsigned int min_idx = 0;
	unsigned int second_idx = 0;
	unsigned int distance;

	// Compare with all previous descriptors
	for (int j = 0; j < num_old_pts; j++) {
		distance = 0;

		// Compute the Hamming distance
		for (int k = 0; k < 4; k++) {
			distance += __popcll(pNewBitString64[k] ^ (*pOldBitString64));	// Note popcll is #ifdef protected to only be called by the NVCC compiler
																			// The NVCC compiler macro CUDACC is only defined while the NVCC compiler is actually running
																			// From MSVS point of view, this function is always #defd out of existence
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

	matches_dist[new_pt_idx * 2 + 0] = min_dist;
	matches_dist[new_pt_idx * 2 + 1] = second_dist;
	matches_idx[new_pt_idx * 2 + 0] = min_idx;
	matches_idx[new_pt_idx * 2 + 1] = second_idx;

}


cudaError_t MyCudaUtils::BruteHammingCUDA(cv::Mat& new_desc, cv::Mat& old_desc, cv::Mat& match_idx, cv::Mat& match_dist) {
	

	unsigned long long* dev_new_desc;
	unsigned long long* dev_old_desc;
	unsigned int* dev_matches_idx;
	float* dev_matches_dist;
	unsigned int* host_matches_idx = new unsigned int[new_desc.rows * 2];
	float * host_matches_dist = new float[new_desc.rows * 2];
	int num_new = new_desc.rows;
	int num_old = old_desc.rows;
	const int BYTESPERDESC = 32;

	try {

		CUDA_CHECK(cudaSetDevice(0));

		CUDA_CHECK(cudaMalloc((void**)&dev_new_desc, new_desc.rows * BYTESIN256BIT));

		CUDA_CHECK(cudaMalloc((void**)&dev_old_desc, old_desc.rows * BYTESIN256BIT));

		CUDA_CHECK(cudaMalloc((void**)&dev_matches_idx, new_desc.rows * 2 * sizeof(unsigned int)));

		CUDA_CHECK(cudaMalloc((void**)&dev_matches_dist, new_desc.rows * 2 * sizeof(float)));

		CUDA_CHECK(cudaMemcpy(dev_new_desc, (unsigned long long*) new_desc.data, num_new * BYTESPERDESC, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(dev_old_desc, (unsigned long long*) old_desc.data, num_old * BYTESPERDESC, cudaMemcpyHostToDevice));

		int blocks = cvCeil(new_desc.rows / 128.0f);
		HammingKernel << < blocks, 128 >> >(dev_new_desc, dev_old_desc, dev_matches_idx, dev_matches_dist, num_new, num_old);

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(host_matches_idx, dev_matches_idx, new_desc.rows * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaMemcpy(host_matches_dist, dev_matches_dist, new_desc.rows * 2 * sizeof(float), cudaMemcpyDeviceToHost));

		unsigned int* pIdx = host_matches_idx;
		float* pDist = host_matches_dist;
		match_dist.create(num_new, 2, CV_32FC1);
		match_idx.create(num_new, 2, CV_32SC1);
		for (int i = 0; i < new_desc.rows; i++) {
			match_dist.at<float>(i, 0) = *(pDist++);
			match_dist.at<float>(i, 1) = *(pDist++);
			match_idx.at<int>(i, 0) = *(pIdx++);
			match_idx.at<int>(i, 1) = *(pIdx++);
		}

		throw(cudaSuccess);

	}
	catch (cudaError_t cudaStatus) {

		cudaFree(dev_new_desc);
		cudaFree(dev_old_desc);
		cudaFree(dev_matches_idx);
		cudaFree(dev_matches_dist);
		delete[] host_matches_idx;
		delete[] host_matches_dist;

		return cudaStatus;
	}

}