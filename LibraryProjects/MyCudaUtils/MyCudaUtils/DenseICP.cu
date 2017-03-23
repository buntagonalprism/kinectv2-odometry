#include "MyCudaUtils.h"
#include "Kernels.h"
#include "Wrappers.h"

__device__ float camPams[CAMPAMSZ];

__device__ int imSizes[IMPAMSZ];

__device__ int pt_matches;

__device__ float imStats[64 * 3];

__device__ unsigned int num_correp[4];
#define TOTAL_IDX 0
#define FOUND_IDX 1
#define FAILED_DIST_IDX 2
#define FAILED_ANGLE_IDX 3

// Project a cloud to 3D
__global__ void ProjectKernel(unsigned short* dimage_new, unsigned short* dimage_old, float* cloud_new, float* cloud_old, int num_pts) {
	
	int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pt_idx >= num_pts)
		return;

	int col = pt_idx % imSizes[DNX];
	int row = roundf((pt_idx - col) / imSizes[DNX]);
	
	unsigned short depth_new = dimage_new[pt_idx];
	unsigned short depth_old = dimage_old[pt_idx];

	// Print statement for debugging to check depth being accessed correctly
	//printf("Row, col: %d,%d Depth_new: %d Depth_old: %d\n", row, col, depth_new, depth_old);

	// Project the point in the new depth image
	if (depth_new > 0) {
		cloud_new[pt_idx * 3 + 2] = depth_new * 0.001f;
		cloud_new[pt_idx * 3 + 0] = static_cast<float> (col - camPams[DCX]) * cloud_new[pt_idx * 3 + 2] * (1 / camPams[DFX]);
		cloud_new[pt_idx * 3 + 1] = static_cast<float> (row - camPams[DCY]) * cloud_new[pt_idx * 3 + 2] * (1 / camPams[DFY]);
		//printf("New Cloud: (row,col) (x,y,z): (%d, %d) (%f,%f,%f)\n", row, col, cloud_new[pt_idx * 3 + 0], cloud_new[pt_idx * 3 + 1], cloud_new[pt_idx * 3 + 2]);
	}
	else {
		cloud_new[pt_idx * 3 + 2] = -1.0f;		// Negative z coordinate to show invalid 
		cloud_new[pt_idx * 3 + 0] = 0.0f;
		cloud_new[pt_idx * 3 + 1] = 0.0f;
	}
	// Print statement for debugging to check point projection is correct
	//printf("New Cloud: (row,col) (x,y,z): (%d, %d) (%f,%f,%f)\n", row, col, cloud_new[pt_idx * 3 + 0], cloud_new[pt_idx * 3 + 1], cloud_new[pt_idx * 3 + 2]);

	// Project the point in the old depth image
	if (depth_old > 0){
		cloud_old[pt_idx * 3 + 2] = depth_old * 0.001f;
		cloud_old[pt_idx * 3 + 0] = static_cast<float> (col - camPams[DCX]) * cloud_old[pt_idx * 3 + 2] * (1 / camPams[DFX]);
		cloud_old[pt_idx * 3 + 1] = static_cast<float> (row - camPams[DCY]) * cloud_old[pt_idx * 3 + 2] * (1 / camPams[DFY]);
	}
	else {
		cloud_old[pt_idx * 3 + 2] = -1.0f;		// Negative z coordinate to show invalid 
		cloud_old[pt_idx * 3 + 0] = 0.0f;
		cloud_old[pt_idx * 3 + 1] = 0.0f;
	}
}

__device__ inline void VecMinus(float vecA[3], float vecB[3], float vec_out[3]) {
	vec_out[0] = vecA[0] - vecB[0];
	vec_out[1] = vecA[1] - vecB[1];
	vec_out[2] = vecA[2] - vecB[2];
}

__device__ inline void VecCross(float vecA[3], float vecB[3], float vec_out[3]) {
	vec_out[0] = vecA[1] * vecB[2] - vecA[2] * vecB[1];
	vec_out[1] = vecA[2] * vecB[0] - vecA[0] * vecB[2];
	vec_out[2] = vecA[0] * vecB[1] - vecA[1] * vecB[0];
}

__device__ inline float VecNorm(float vec[3]) {
	return sqrtf(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

__device__ inline void VecNormalise(float vec_in[3], float vec_out[3]) {
	float norm = VecNorm(vec_in);
	vec_out[0] = vec_in[0] / norm;
	vec_out[1] = vec_in[1] / norm;
	vec_out[2] = vec_in[2] / norm;
}

__device__ inline float VecDot(float vecA[3], float vecB[3]) {
	return vecA[0] * vecB[0] + vecA[1] * vecB[1] + vecA[2] * vecB[2];
}

// Cloud is required to be stored row major order of points i.e. for Pt_Row,Col = (x,y,z); cloud =  [Pt_0,0, Pt_0,1, Pt_0,2,...Pt_0,n, Pt_1,, 
__global__ void ComputeNormalsKernel(float* cloud, float* normals, int num_pts) {
	
	int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pt_idx >= num_pts)
		return;

	int num_cols = imSizes[DNX];
	int num_rows = imSizes[DNY];

	int col = pt_idx % imSizes[DNX];
	int row = roundf((pt_idx - col) / imSizes[DNX]);


	float pt[3] = { 0.0 };
	float pt_1col[3] = { 0.0 };   // Offset from pt by one column
	float pt_1row[3] = { 0.0 };   // Offset from pt by one row

	pt[0] = cloud[pt_idx * 3 + 0];
	pt[1] = cloud[pt_idx * 3 + 1];
	pt[2] = cloud[pt_idx * 3 + 2];

	bool normError = false;
	// Check whether we are at the border
	if ((col == (num_cols - 1)) || (row == (num_rows - 1)))
		normError = true;

	if (!normError) {
		pt_1col[0] = cloud[(pt_idx + 1) * 3 + 0];
		pt_1col[1] = cloud[(pt_idx + 1) * 3 + 1];
		pt_1col[2] = cloud[(pt_idx + 1) * 3 + 2];

		pt_1row[0] = cloud[(pt_idx + num_cols) * 3 + 0];
		pt_1row[1] = cloud[(pt_idx + num_cols) * 3 + 1];
		pt_1row[2] = cloud[(pt_idx + num_cols) * 3 + 2];
	}

	// Check the three have valid depth readings
	if (pt[2] < 0 || pt_1col[2] < 0 || pt_1row[2] < 0)
		normError = true;

	// Compute normal through local vector cross product
	if (!normError) {
		float vecRow[3], vecCol[3], normal[3], norm_normal[3];
		VecMinus(pt_1col, pt, vecCol);
		VecMinus(pt_1row, pt, vecRow);
		VecCross(vecCol, vecRow, normal);

		VecNormalise(normal, norm_normal);

		normals[pt_idx * 3 + 0] = norm_normal[0];
		normals[pt_idx * 3 + 1] = norm_normal[1];
		normals[pt_idx * 3 + 2] = norm_normal[2];
	}

	// Set normal to greater than unit magnitude to show error computing normal
	if (normError) {
		normals[pt_idx * 3 + 0] = 2.0f;
		normals[pt_idx * 3 + 1] = 2.0f;
		normals[pt_idx * 3 + 2] = 2.0f;
	}

}

// Finds a correspondance in the new cloud for every point in the old cloud
__global__ void ProjectiveAssociationKernel(float* trans_cloud_old, float* trans_normals_old, float* cloud_new, float* normals_new, int* correp_mask, int num_pts) {

	int pt_old_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pt_old_idx >= num_pts)
		return;

	int num_cols = imSizes[DNX];
	int num_rows = imSizes[DNY];

	float old_point[3], new_point[3], old_normal[3], new_normal[3];
	old_point[0] = trans_cloud_old[pt_old_idx * 3 + 0];
	old_point[1] = trans_cloud_old[pt_old_idx * 3 + 1];
	old_point[2] = trans_cloud_old[pt_old_idx * 3 + 2];

	old_normal[0] = trans_normals_old[pt_old_idx * 3 + 0];
	old_normal[1] = trans_normals_old[pt_old_idx * 3 + 1];
	old_normal[2] = trans_normals_old[pt_old_idx * 3 + 2];

	bool correp_valid = true;

	// Check old point has valid depth and old normal is valid unit vector
	if (old_point[2] < 0 || old_normal[0] > 1.0)
		correp_valid = false;

	// Find where the transformed old 3D point projects to in new frame
	int col_proj_new, row_proj_new;
	if (correp_valid) {
		float invZ = 1.0 / old_point[2];
		 col_proj_new = roundf((old_point[0] * camPams[DFX] * invZ) + camPams[DCX]);
		 row_proj_new = roundf((old_point[1] * camPams[DFY] * invZ) + camPams[DCY]);

		 // Check reprojection falls into valid image coordinates
		 if (col_proj_new < 0 || row_proj_new < 0 || col_proj_new >= num_cols || row_proj_new >= num_rows)
			 correp_valid = false;
	}
	
	// Get the new point and normal
	int pt_new_idx;
	if (correp_valid){
		pt_new_idx = col_proj_new + row_proj_new*num_cols;

		new_point[0] = cloud_new[pt_new_idx * 3 + 0];
		new_point[1] = cloud_new[pt_new_idx * 3 + 1];
		new_point[2] = cloud_new[pt_new_idx * 3 + 2];

		new_normal[0] = normals_new[pt_new_idx * 3 + 0];
		new_normal[1] = normals_new[pt_new_idx * 3 + 1];
		new_normal[2] = normals_new[pt_new_idx * 3 + 2];

		// Check the new point and normal is valid
		if (new_point[2] < 0 || new_normal[0] > 1.0)
			correp_valid = false;
		else
			atomicAdd(&num_correp[FOUND_IDX], 1);
	}

	// Check for valid correspondance by euclidean distance and angle thresholds
	if (correp_valid) {
		float distVec[3];
		VecMinus(new_point, old_point, distVec);
		float dist = VecNorm(distVec);

		
		float angle = fabsf(VecDot(new_normal, old_normal));

		if (dist > 0.1) {
			correp_valid = false;
			atomicAdd(&num_correp[FAILED_DIST_IDX], 1);
			//if (pt_old_idx == 102650 || pt_old_idx == 102660) {
			//	printf("Old Pt %d trans to: (%.3f, %.3f, %.3f), projects to: (%d, %d) with correp new point (%.3f, %.3f, %.3f) and dist %f\n", pt_old_idx, old_point[0], old_point[1], old_point[2], col_proj_new, row_proj_new,
			//		new_point[0], new_point[1], new_point[2], dist);
			//}
			// Print statements for debugging to check projective association validity
			
		}
		//else {
		//	printf("Successful correspondance");
		//}
	}

	// Mark a valid correspondance, or not
	if (correp_valid) {
		correp_mask[pt_old_idx] = pt_new_idx;
		atomicAdd(&num_correp[TOTAL_IDX], 1);
	}
	else {
		correp_mask[pt_old_idx] = -1;
	}
}


__global__ void ComputeErrorFunc(float* source_old, float *dest_new, float *normals_new, int* matches_mask, float* A_mat, float* B_mat) {

	int col_idx = threadIdx.x + blockIdx.x * blockDim.x;

	int num_cols = imSizes[DNX];
	int num_rows = imSizes[DNY];

	int src_pt_idx = 0, dest_pt_idx = 0;
	float src_pt[3], dest_pt[3], norm[3], cross[3], diff[3], dot;

	float BMat_sum[6] = { 0.0 };
	float AMat_sum[36] = { 0.0 };

	int matches_this_col = 0;

	for (int row = 0; row < num_rows; row++) {
		src_pt_idx = col_idx + row * num_cols;
		dest_pt_idx = matches_mask[src_pt_idx];
		if (dest_pt_idx == -1)
			continue;
		matches_this_col++;
		src_pt[0] = source_old[src_pt_idx * 3 + 0];
		src_pt[1] = source_old[src_pt_idx * 3 + 1];
		src_pt[2] = source_old[src_pt_idx * 3 + 2];

		dest_pt[0] = dest_new[dest_pt_idx * 3 + 0];
		dest_pt[1] = dest_new[dest_pt_idx * 3 + 1];
		dest_pt[2] = dest_new[dest_pt_idx * 3 + 2];

		norm[0] = normals_new[dest_pt_idx * 3 + 0];
		norm[1] = normals_new[dest_pt_idx * 3 + 1];
		norm[2] = normals_new[dest_pt_idx * 3 + 2];

		VecCross(src_pt, norm, cross);
		VecMinus(src_pt, dest_pt, diff);
		dot = VecDot(diff, norm);

		// Add to the B matrix
		for (int i = 0; i < 3; i++) {
			BMat_sum[i] -= cross[i] * dot;
			BMat_sum[i + 3] -= norm[i] * dot;
		}

		// Storing column-major
		for (int i = 0; i < 3; i++) {
			float multTop = cross[i];
			float multBot = norm[i];
			for (int j = 0; j < 3; j++) {
				AMat_sum[i + 6 * j] += multTop*cross[j];
				AMat_sum[i + 6 * (j + 3)] += multTop*norm[j];

				AMat_sum[i + 3 + 6 * j] += multBot*cross[j];
				AMat_sum[i + 3 + 6 * (j + 3)] += multBot*norm[j];
			}
		}
		// Print statements for debugging to check the computation of the error function
		//if (col_idx == 200 && (matches_this_col == 170 || matches_this_col == 171)) {
		//	printf("Src_pt: %f, %f, %f\n", src_pt[0], src_pt[1], src_pt[2]);
		//	printf("Dest_pt:  %f, %f, %f\n", dest_pt[0], dest_pt[1], dest_pt[2]);
		//	printf("Norm: %f, %f, %f\n", norm[0], norm[1], norm[2]);
		//	printf("Cross: %f, %f, %f\n", cross[0], cross[1], cross[2]);
		//	printf("Dot: %f\n", dot);
		//	printf("BMat: \n");
		//	for (int i = 0; i < 6; i++) {
		//		printf("%f\n", BMat_sum[i]);
		//	}
		//	printf("\nAmat:\n");
		//	for (int i = 0; i < 6; i++) {
		//		for (int j = 0; j < 6; j++) {
		//			printf("%f, ", AMat_sum[j * 6 + i]);
		//		}
		//		printf("\n");
		//	}
		//	printf("\n");
		//}
	}

	// Copy back to global array
	for (int i = 0; i < 36; i++) {
		A_mat[col_idx * 36 + i] = AMat_sum[i];
	}

	for (int i = 0; i < 6; i++) {
		B_mat[col_idx * 6 + i] = BMat_sum[i];
	}


}

void RfromT(float* T_in, float* R_out) {
	for (int i = 0; i < 16; i++) {
		R_out[i] = T_in[i];
	}
	R_out[12] = R_out[13] = R_out[14] = 0.0f;
}

void MatMult(float* M1, float* M2, float* M_out, int rows1, int cols1, int cols2) {
	float sum = 0;
	for (int c = 0; c < rows1; c++)
	{
		for (int d = 0; d < cols2; d++)
		{
			for (int k = 0; k < cols1; k++)
			{
				sum = sum + M1[k*cols1 + c] * M2[d*cols2 + k];
			}

			M_out[d*cols2 + c] = sum;
			sum = 0;
		}
	}
}

// THIS FUNCTION IS HARD CODED. USE WITH CAUTION
__global__ void ComputeDepthImStats(unsigned short* dimage) {
	int col_idx = threadIdx.x;
	int start_pt = blockIdx.y * 8 * 512 + blockIdx.x * 64 + col_idx;

	__shared__ float sumArray[64];
	__shared__ float varArray[64];
	__shared__ int validArray[64];
	__shared__ float mean;
	__shared__ int valid;
	sumArray[col_idx] = 0.0;
	varArray[col_idx] = 0.0;
	validArray[col_idx] = 0;
	for (int i = 0; i < 53; i++) {
		unsigned short d = dimage[start_pt + 512 * i];
		if (d > 0) {
			sumArray[col_idx] += (float)dimage[start_pt + 512 * i];
			validArray[col_idx]++;
		}
	}

	__syncthreads();

	if (col_idx == 0) {
		float sum = 0;
		valid = 0;
		for (int i = 0; i < 64; i++) {
			sum += sumArray[i];
			valid += validArray[i];
		}
		if (valid == 0) mean = 0;
		else mean = sum / (valid);
	}

	__syncthreads();
	for (int i = 0; i < 53; i++) {
		unsigned short d = dimage[start_pt + 512 * i];
		if (d > 0) {
			varArray[col_idx] += (d - mean)* (d - mean);
		}
	}
	__syncthreads();

	if (col_idx == 0) {
		float var = 0;
		for (int i = 0; i < 64; i++) var += varArray[i];
		if (valid == 0) var = 0;
		else var /= (valid);
		imStats[blockIdx.x * 3 + blockIdx.y * 8 * 3 + 0] = mean;
		imStats[blockIdx.x * 3 + blockIdx.y * 8 * 3 + 1] = var;
		imStats[blockIdx.x * 3 + blockIdx.y * 8 * 3 + 2] = valid;
	}

}

void MakeTransIdentity(float* T) {
	for (int i = 0; i < 16; i++) { T[i] = 0; }
	T[0] = T[5] = T[10] = T[15] = 1.0;
}

cudaError_t MyCudaUtils::DenseICP(cv::Mat& dimage_new, cv::Mat& dimage_old, float* H_camPams, int* H_imSizes, float* T_hint, float depthStats[128]) {

	assert(dimage_old.type() == CV_16U && dimage_new.type() == CV_16U);

	assert(dimage_old.size == dimage_new.size);

	int num_pts = dimage_new.rows * dimage_new.cols;

	bool drawDepthMaps = false;
	if (drawDepthMaps) DrawDepthMaps(dimage_new, dimage_old);

	// Since cv::Mat is row major this actually performs the same job as column-major
	// points-in-columns eigen matrix
	cv::Mat normals_new(num_pts, 3, CV_32F);

	cv::Mat cloud_new(num_pts, 3, CV_32F);

	cv::Mat cloud_old(num_pts, 3, CV_32F);

	cv::Mat mapped_cloud_old(num_pts, 3, CV_32F);

	cv::Mat correp_mask(dimage_new.rows, dimage_new.cols, CV_32S);

	// Device memory pointers
	float *D_cloud_new, *D_cloud_old;

	unsigned short *D_dimage_new, *D_dimage_old;

	float *D_normals_new, *D_normals_old;

	float *D_trans, *D_rot;

	float *D_mapped_cloud_old, *D_mapped_normals_old;

	int *D_correp_mask;

	float *D_A_mat, *D_B_mat;

	// Output transform
	float T_curr[16] = { 0.0 };

	

	bool drawTransCloud = false;
	bool drawTransNormals = false;
	bool drawInitProjection = false;
	bool drawInitNormals = false;
	bool drawCorrep = false;
	

	try {

		// Allocate for the clouds
		CUDA_MALLOC(D_cloud_new, num_pts * 3 * sizeof(float));
		CUDA_MALLOC(D_cloud_old, num_pts * 3 * sizeof(float));


		// For the correspondance mask
		CUDA_MALLOC(D_correp_mask, num_pts * sizeof(int));

		// Copy across the depth images
		CUDA_INIT_MEM(D_dimage_new, dimage_new.data, num_pts * sizeof(unsigned short));
		CUDA_INIT_MEM(D_dimage_old, dimage_old.data, num_pts * sizeof(unsigned short));


		// Allocate for the normal map
		CUDA_MALLOC(D_normals_new, num_pts * 3 * sizeof(float));
		CUDA_MALLOC(D_normals_old, num_pts * 3 * sizeof(float));


		// For old cloud and normals transformed into estimated new camera space
		CUDA_MALLOC(D_mapped_cloud_old, num_pts * 3 * sizeof(float));
		CUDA_MALLOC(D_mapped_normals_old, num_pts * 3 * sizeof(float));


		// Camera parameters for projection
		CUDA_CHECK(cudaMemcpyToSymbol(camPams, H_camPams, CAMPAMSZ * sizeof(float)));
		CUDA_CHECK(cudaMemcpyToSymbol(imSizes, H_imSizes, IMPAMSZ * sizeof(int)));


		// A and B matrices summed across each column by a single GPU thread
		CUDA_MALLOC(D_A_mat, dimage_new.cols * 36 * sizeof(float));
		CUDA_MALLOC(D_B_mat, dimage_new.cols * 6 * sizeof(float));

		// Perform the projection of both clouds
		int blocks = cvCeil(num_pts / 128.0f);
		ProjectKernel << < blocks, 128 >> >(D_dimage_new, D_dimage_old, D_cloud_new, D_cloud_old, num_pts);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// For debugging - check the projection of the dense clouds
		if (drawInitProjection) {
			CUDA_DOWNLOAD(cloud_new.data, D_cloud_new, num_pts*sizeof(float)* 3);
			CUDA_DOWNLOAD(cloud_old.data, D_cloud_old, num_pts*sizeof(float)* 3);
			DrawClouds((float *)cloud_new.data, num_pts, (float*)cloud_old.data, 0.8, 3.0);
		}

		// Compute the normals
		ComputeNormalsKernel << < blocks, 128 >> >(D_cloud_new, D_normals_new, num_pts);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		ComputeNormalsKernel << < blocks, 128 >> >(D_cloud_old, D_normals_old, num_pts);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// For debugging, check the computation of the normals
		if (drawInitNormals) {
			CUDA_DOWNLOAD(normals_new.data, D_normals_new, num_pts * 3 * sizeof(float));
			DrawNormals((float*) cloud_new.data, num_pts, (float*) normals_new.data, 0.08, 0.26);
		}

		// Compute statistics for the depth image. For an equally sized 8x8 grid
		// compute the mean and std_dev of the depth pixels
		ComputeDepthImStats << <dim3(8, 8, 1), 64 >> >(D_dimage_new);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpyFromSymbol(depthStats, imStats, 192 * sizeof(float)));

		// Perform the initial transformation of old points and normals to new frame
		//T_hint[12] = -0.02;
		//T_hint[13] = -0.02;
		//T_hint[14] = 0.02;
		CUDA_INIT_MEM(D_trans, T_hint, 16 * sizeof(float));
		float R[16];
		RfromT(T_hint, R);
		CUDA_INIT_MEM(D_rot, R, 16 * sizeof(float));

		
		float deltaX = 100;
		float stateVec[6] = { 10.0 };
		
		for (int i = 0; i< 16; i++) T_curr[i] = T_hint[i];
		int iter = 0;
		char file[256];
		// Begin the loop of iterating to completion
		int maxIter = 30;
		bool failed_hint = false;
		while (deltaX > 1e-8 && iter++ < maxIter) {
			// Transform old points and normals by current estimate
			TransformKernel<<< blocks, 128 >>>(D_cloud_old, D_trans, D_mapped_cloud_old, num_pts);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			TransformKernel <<< blocks, 128 >>>(D_normals_old, D_rot, D_mapped_normals_old, num_pts);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			// For debugging, draw transformed old cloud and the two on top of each other
			if (drawTransCloud) {
				CUDA_DOWNLOAD(mapped_cloud_old.data, D_mapped_cloud_old, num_pts*sizeof(float)* 3);
				CUDA_DOWNLOAD(cloud_new.data, D_cloud_new, num_pts *sizeof(float)* 3);
				sprintf_s(file, "Iter%d.png", iter-1);
				DrawClouds((float*)cloud_new.data, num_pts, (float*)mapped_cloud_old.data, 0.8, 3.0, file);

				// For saving screenshots
				//sprintf_s(file, "Iter%d.png", iter++);
				
			}
			// For debugging, draws the transformed normals 
			if (drawTransNormals) {
				CUDA_DOWNLOAD(normals_new.data, D_mapped_normals_old, num_pts * 3 * sizeof(float));
				DrawNormals((float*)cloud_new.data, num_pts, (float*)normals_new.data, 0.08, 0.26);
			}
			
			
			// Get the correspondance map through projective data assocation
			int host_num_correp[4] = { 0 };
			CUDA_CHECK(cudaMemcpyToSymbol(num_correp, &host_num_correp, 4*sizeof(int)));
			ProjectiveAssociationKernel << <blocks, 128 >> >(D_mapped_cloud_old, D_mapped_normals_old, D_cloud_new, D_normals_new, D_correp_mask, num_pts);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaMemcpyFromSymbol( &host_num_correp, num_correp, 4*sizeof(int)));

			if (drawCorrep ) {
				CUDA_DOWNLOAD(mapped_cloud_old.data, D_mapped_cloud_old, num_pts*sizeof(float)* 3);
				CUDA_DOWNLOAD(cloud_new.data, D_cloud_new, num_pts *sizeof(float)* 3);
				CUDA_DOWNLOAD(cloud_old.data, D_cloud_old, num_pts*sizeof(float)* 3);
				CUDA_DOWNLOAD(correp_mask.data, D_correp_mask, num_pts * sizeof(int));
				DrawCorrep((float*)mapped_cloud_old.data, (float*)cloud_new.data, num_pts, (int*)correp_mask.data);
			}

			// Hint might be bad, try restarting with identity
			if (host_num_correp[0] < 100 && failed_hint == false) {
				failed_hint = true;
				MakeTransIdentity(T_curr);
				CUDA_UPLOAD(D_trans, T_curr, 16 * sizeof(float));
				float R[16];
				RfromT(T_hint, R);
				CUDA_UPLOAD(D_rot, R, 16 * sizeof(float));
				continue;
			}
			// Hint and identity failed, ICP has failed and return identity
			else if (host_num_correp[0] < 100 && failed_hint == true) {
				MakeTransIdentity(T_curr);
				break;
			}

			
			//QueryPoint( correp_mask);
			

			// Compute the Ax - b = 0 error function
			ComputeErrorFunc << <16, 32 >> >(D_mapped_cloud_old, D_cloud_new, D_normals_new, D_correp_mask, D_A_mat, D_B_mat);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			float* H_A = new float[dimage_new.cols * 36];
			float* H_B = new float[dimage_new.cols * 6];
			CUDA_DOWNLOAD(H_A, D_A_mat, dimage_new.cols * 36 * sizeof(float));
			CUDA_DOWNLOAD(H_B, D_B_mat, dimage_new.cols * 6 * sizeof(float));


			// Solve 
			float newStateVec[6];
			FindMinimisingTransform(H_A, H_B, dimage_new.cols, newStateVec);
			delete[] H_A;
			delete[] H_B;

			// Compute amount of change in x - break if small
			deltaX = 0.0;
			for (int i = 0; i < 6; i++)
				deltaX += (newStateVec[0])*(newStateVec[0]);
			for (int i = 0; i < 6; i++) stateVec[i] = newStateVec[i];
			
			// Compute equivalent transform and compound with existing estimate
			float alp = stateVec[0], bet = stateVec[1], gam = stateVec[2];
			float tx = stateVec[3], ty = stateVec[4], tz = stateVec[5];
			float T[16] = { 0.0 };
			T[0] = cos(gam)*cos(bet);
			T[1] = sin(gam)*cos(bet);
			T[2] = -sin(bet);

			T[4] = -sin(gam)*cos(alp) + cos(gam)*sin(bet)*sin(alp);
			T[5] = cos(gam)*cos(alp) + sin(gam)*sin(bet)*sin(alp);
			T[6] = cos(bet)*sin(alp);

			T[8] = sin(gam)*sin(alp) + cos(gam)*sin(bet)*cos(alp);
			T[9] = -cos(gam)*sin(alp) + sin(gam)*sin(bet)*cos(alp);
			T[10] = cos(bet)*cos(alp);

			T[12] = tx; T[13] = ty; T[14] = tz;
			T[15] = 1.0;

			float T_temp[16];
			MatMult(T_curr, T, T_temp, 4, 4, 4);
			for (int i = 0; i < 16; i++) T_curr[i] = T_temp[i];
			CUDA_UPLOAD(D_trans, T_curr, 16 * sizeof(float));
			float R[16];
			RfromT(T_hint, R);
			CUDA_UPLOAD(D_rot, R, 16 * sizeof(float));


		}
		// Did not converge, return Identity
		if (iter == maxIter) {
			for (int i = 0; i < 16; i++) { T_curr[i] = 0; }
			T_curr[0] = T_curr[5] = T_curr[10] = T_curr[15] = 1.0;
		}
			 
		throw(cudaSuccess);

	}
	catch (cudaError_t cudaStatus) {
		cudaFree(D_cloud_new);
		cudaFree(D_cloud_old);
		cudaFree(D_dimage_new);
		cudaFree(D_dimage_old);
		cudaFree(D_normals_new);
		cudaFree(D_normals_old);
		cudaFree(D_mapped_cloud_old);
		cudaFree(D_mapped_normals_old);
		cudaFree(D_correp_mask);
		cudaFree(D_trans);
		cudaFree(D_rot);
		cudaFree(D_A_mat);
		cudaFree(D_B_mat);

		for (int i = 0; i < 16; i++) T_hint[i] = T_curr[i];

		return cudaStatus;
	}


	// Create normal map for the new depth image

	// Associate points by 



}