
#ifndef KERNELS_H
#define KERNELS_H
#include "MyCudaUtils.h"


__global__ void HammingKernel(unsigned long long* new_desc, unsigned long long* old_desc, unsigned int* matches_idx, float* matches_dist, int num_new_pts, int num_old_pts);

__global__ void StarDetectorKernel(int* intImage, int* tiltImage, int* flatTiltImage, int numSizes, int* sizes, float* invAreas, int* starCnrs, int numPairs, int* pairs, int rows, int cols, int border, float* responses_out, short* sizes_out);

__global__ void MapDepthToColourKernel(unsigned short* dImage, unsigned short* mapped, float* camPams, int* imSizes);

// Transforms a point cloud by a transform T
// [in]  Input cloud stored [x1,y1,z1, x2,y2,z2, x3,y3,z3...]
// [in]  Transform to apply, stored COLUMN-MAJOR, i.e. for T_Row,Col  T = [T_0,0, T_1,0, T_2,0, T_3,0, T_0,1, T_1,1, T_2,1
__global__ void TransformKernel(float* in, float* T, float* out, int num_pts);


// ICP Device Kernel Declarations
__global__ void ComputeErrorFunc(float* source_old, float *dest_new, float *normals_new, int* matches_mask, float* A_mat, float* B_mat);
void RunComputeErrorFunc(float* source_old, float *dest_new, float *normals_new, int* matches_mask, float* A_mat, float* B_mat);

__global__ void ProjectiveAssociationKernel(float* trans_cloud_old, float* trans_normals_old, float* cloud_new, float* normals_new, int* correp_mask, int num_pts);

__global__ void ComputeNormalsKernel(float* cloud, float* normals, int num_pts);

__global__ void ProjectKernel(unsigned short* dimage_new, unsigned short* dimage_old, float* cloud_new, float* cloud_old, int num_pts);

#endif