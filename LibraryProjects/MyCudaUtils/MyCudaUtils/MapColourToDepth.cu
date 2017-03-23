



#include "MyCudaUtils.h"



__global__ void MapColourToDepthKernel(unsigned short *depthImage_in, unsigned char * colourImage_in, unsigned char* mapped_out) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	
	int ndrows = 424;
	int ndcols = 512;
	int nccols = 1920;
	int ncrows = 1080;
	if (!(row < ndrows))
		return;

	float depth_fx = 361.41463;
	float depth_fy = 361.13281;
	float depth_cx = 250.92297;
	float depth_cy = 203.69426;
	float colour_fx = 1064.00189;
	float colour_fy = 1063.74614;
	float colour_cx = 946.74256;
	float colour_cy = 539.82995;

	float deltaX = 0.0522;
	float deltaZ = 0.0059;

	float depth_fx_inv = 1 / depth_fx;
	float depth_fy_inv = 1 / depth_fy;
	float inv_z_c = 0.0f;

	// 3D coordinates 
	float x_d, y_d, z_d, x_c, y_c, z_c;

	// Pixel coordinates (v,u) is (row, column)
	int v_c, u_c;


	unsigned short* pDepthRow = depthImage_in + (row*ndcols);
	int d; bool valid = false;
	for (int col = 0; col < ndcols; col++){
		d = *pDepthRow++;
		valid = (d != -0);
		//mapped_out[row*ndcols + col] = row+col;
		
		if (valid) {
			z_d = 0.001*(float)d;
			x_d = (col - depth_cx) * depth_fx_inv * z_d;
			y_d = (row - depth_cy) * depth_fy_inv * z_d;

			// Apply the offset between them
			x_c = x_d + deltaX;
			z_c = z_d + deltaZ;
			y_c = y_d;				// Negligible y offset

			inv_z_c = 1 / z_c;		// To avoid performing two divisions

			// Project back to image coordinates (no round function in VS2012 cmath or math.h
			u_c = static_cast<int>(colour_cx + (colour_fx * x_c * inv_z_c) + 0.5);
			v_c = static_cast<int>(colour_cy + (colour_fy * y_c * inv_z_c) + 0.5);

			// Check the point lies in the colour image
			if (u_c > (nccols - 1) || u_c < 0 || v_c >(ncrows - 1) || v_c < 0)
				valid = false;
		}
		// Get the colour / intensity at that point and insert in output mapped image
		if (valid)
			mapped_out[row*ndcols + col] = colourImage_in[v_c*nccols + u_c];

		else
			mapped_out[row*ndcols + col] = 0;
	}
}


cudaError_t MyCudaUtils::MapColourToDepth(cv::Mat& cimage_in, cv::Mat& dimage_in, cv::Mat& mapped_out, bool singleBlock) {

	mapped_out.create(424, 512, CV_8UC1);

	unsigned short* dev_dimage_in;
	unsigned char* dev_cimage_in;
	unsigned char* dev_mapped_out;
	int dsize = 512 * 424;
	int csize = 1920 * 1080;

	try {

		CUDA_CHECK(cudaSetDevice(0));

		CUDA_CHECK(cudaMalloc((void**)&dev_dimage_in, dsize * sizeof(unsigned short)));

		CUDA_CHECK(cudaMalloc((void**)&dev_cimage_in, csize * sizeof(unsigned char)));

		CUDA_CHECK(cudaMalloc((void**)&dev_mapped_out, dsize * sizeof(unsigned char)));

		CUDA_CHECK(cudaMemcpy(dev_dimage_in, dimage_in.data, dsize * sizeof(unsigned short), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(dev_cimage_in, cimage_in.data, csize * sizeof(unsigned char), cudaMemcpyHostToDevice));

		MapColourToDepthKernel << < 16, 32 >> >(dev_dimage_in, dev_cimage_in, dev_mapped_out);

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(mapped_out.data, dev_mapped_out, dsize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		throw (cudaSuccess);
	}
	catch (cudaError_t cudaStatus) {
		
		cudaFree(dev_dimage_in);
		cudaFree(dev_cimage_in);
		cudaFree(dev_mapped_out);

		return cudaStatus;
	}

}

