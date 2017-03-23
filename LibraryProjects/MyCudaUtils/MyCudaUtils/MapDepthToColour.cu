

#include "MyCudaUtils.h"






__global__ void MapDepthToColourKernel(unsigned short* dImage, unsigned short* mapped, float* camPams, int* imSizes) {

	int pt_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (pt_idx >= imSizes[DNX] * imSizes[DNY])
		return;

	int col = pt_idx % imSizes[DNX];
	int row = roundf( (pt_idx - col) / imSizes[DNX]);

	unsigned short depth = dImage[pt_idx];
	if (depth == 0)
		return;
 
	float out[3] = { 0.0 };
	float in[3] = { 0.0 };

	// Project the point
	in[2] = depth * 0.001f;
	in[0] = static_cast<float> (col - camPams[DCX]) * in[2] * (1/camPams[DFX]);
	in[1] = static_cast<float> (row - camPams[DCY]) * in[2] * (1/camPams[DFY]);

	// Transform the point
	float* T = &camPams[TRANS];
	out[0] = T[0] * in[0] + T[4] * in[1] + T[8] * in[2] + T[12];	// R00*xi + R01*yi + R02*zi + t_x  
	out[1] = T[1] * in[0] + T[5] * in[1] + T[9] * in[2] + T[13];	// R10*xi + R11*yi + R12*zi + t_y  
	out[2] = T[2] * in[0] + T[6] * in[1] + T[10] * in[2] + T[14];	// R20*xi + R21*yi + R22*zi + t_z 

	//printf("Px: %d,%d In: %f,%f,%f Out: %f,%f,%f\n", row, col, in[0], in[1], in[2], out[0], out[1], out[2]);

	// Project it back 
	float invZ = 1.0 / out[2];
	col = roundf((out[0] * camPams[CFX] * invZ) + camPams[CCX]);
	row = roundf((out[1] * camPams[CFY] * invZ) + camPams[CCY]);


	// Check boundaries and write to the mapped depth image
	if (row < imSizes[CNY] && row >= 0 && col >= 0 && col < imSizes[CNX]) {
		mapped[col + row * imSizes[CNX]] = (unsigned short)(out[2] * 1000);
	}
	// Some way to output how many points we successfully map?

}



cudaError_t MyCudaUtils::MapDepthToColour(cv::Mat& dImage, cv::Mat& mImage, float* camPams, int* imSizes) {

	mImage = cv::Mat::zeros(imSizes[CNY], imSizes[CNX], CV_16UC1);

	unsigned short* D_dimage; int depthSize = dImage.rows * dImage.cols * sizeof(unsigned short);
	unsigned short* D_mimage; int mappedSize = mImage.rows * mImage.cols * sizeof(unsigned short);
	float* D_camPams; int camSize = (8 + 16) * sizeof(float);
	int* D_imPams; int imSize = 4 * sizeof(int);

	try {
		CUDA_INIT_MEM(D_dimage, dImage.data, depthSize);

		CUDA_INIT_MEM(D_camPams, camPams, camSize);

		CUDA_INIT_MEM(D_imPams, imSizes, imSize);

		CUDA_INIT_MEM(D_mimage, mImage.data, mappedSize);

		int blocks = cvCeil(dImage.rows * dImage.cols / 128.0f);
		MapDepthToColourKernel << <blocks, 128 >> >(D_dimage, D_mimage, D_camPams, D_imPams);

		CUDA_CHECK(cudaMemcpy(mImage.data, D_mimage, mappedSize, cudaMemcpyDeviceToHost));

		throw (cudaSuccess);
	}
	catch (cudaError_t cudaStatus) {
	
		cudaFree(D_dimage);
		cudaFree(D_mimage);
		cudaFree(D_camPams);
		cudaFree(D_imPams);
		return cudaStatus;
	}

}