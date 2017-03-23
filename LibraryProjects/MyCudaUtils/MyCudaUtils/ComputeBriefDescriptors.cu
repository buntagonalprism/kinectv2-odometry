#include "MyCudaUtils.h"

// A region of interest which contains a rectangle (and is constructed with a rectangle). 
// Turns out this is really only used for the keypoint near border removal, as a quick check of 
// std::remove_if in the runByImageBorder below says that a predicate is a required input.
// The predicate is a function (or in this case the operator() of the object) which returns
// boolean true/false when called with a single element from the vector as to whether it 
// should be removed or not. remove_if internally calls the RoiPredicate operator on 
// each element of the keypoint, and it returns a boolean if the keypoint location 
// is within the rectangle. remove_if then returns a vector of booleans which the 
// std::vector.erase function uses to determine which keypoints to remove. Nifty
struct RoiPredicate
{
	RoiPredicate(const cv::Rect& _r) : r(_r)
	{}

	bool operator()(const cv::KeyPoint& keyPt) const
	{
		return !r.contains(keyPt.pt);
	}

	cv::Rect r;
};

// Function removes keypoints near the boarder of the image within a particular border size
// This shouldn't be too much of a problem, I believe that STAR doesn't find them too close either
void runByImageBorder(std::vector<cv::KeyPoint>& keypoints, cv::Size imageSize, int borderSize)
{
	if (borderSize > 0)
	{
		if (imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2)
			keypoints.clear();
		else
			keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(),
			RoiPredicate(cv::Rect(cv::Point(borderSize, borderSize),
			cv::Point(imageSize.width - borderSize, imageSize.height - borderSize)))),
			keypoints.end());
	}
}



// Device equivalent function for computing the smoothed function
// [in]  Ptr to the integral image
// [in]  The x and y coordinates of the keypoint the current CUDA thread is operating on
// [in]  The relative position around this keypoint we are querying as a point for our Brief binary descriptor
// [in]  The stride of the integral image so we can access elements without having nice row, column matrix 
__device__ int smoothedSumDevice(int* intImage, int pty, int ptx, int rely, int relx, int stride) {

	const int KERNEL_SIZE = 9;
	const int HALF_KERNEL = KERNEL_SIZE / 2;

	int img_y = (int)(pty + 0.5) + rely;
	int img_x = (int)(ptx + 0.5) + relx;	

	return   intImage[(img_y + HALF_KERNEL + 1)*stride + img_x + HALF_KERNEL + 1]
		- intImage[(img_y + HALF_KERNEL + 1)*stride + img_x - HALF_KERNEL]
		- intImage[(img_y - HALF_KERNEL)*stride + img_x + HALF_KERNEL + 1]
		+ intImage[(img_y - HALF_KERNEL)*stride + img_x - HALF_KERNEL];

	// Smooths by computing a box filter - average of all the surrounding pixels
	// which of course requires their sum. We compute this efficiently by constructing the
	// integral image and taking the four corner points around our kernel. Here we use a
	// kernel size of 9 - i.e. we smooth over a window 9 pix across to get the intensity value 
}



// Kernel for computing the descriptor for a single keypoint. 
// We could probably be more efficient with global memory accesses because there's a lot per thread
// [in]  A pointer to the row-major stored integral image data
// [in]  The stride of the integral image in pixels (also bytes for a uchar image)
// [in]  The set of keypoints laid out with all x-coords in first row, all y-coords in second row
// [in]  The number of points N in the set of keypoints
// [out] The 256bit descriptor array, of size 32*N bytes 
__global__ void pixelTest32Kernel(int* intImage, int imStride, float* kps, int num_pts, unsigned char* descriptors) {

	int pt_idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (!(pt_idx < num_pts))										// Check thread index is valid 
		return;
	int ptx = kps[pt_idx];											// X-coords on first row of matrix
	int pty = kps[pt_idx + num_pts];								// Y-coords on second row
	unsigned char* desc = descriptors + (32 * pt_idx);				// Where to put the descriptor for this keypoint in output array
#define DEVSMOOTHED(y,x) smoothedSumDevice(intImage, pty, ptx, y, x, imStride)
	desc[0] = (uchar)(((DEVSMOOTHED(-2, -1) < DEVSMOOTHED(7, -1)) << 7) + ((DEVSMOOTHED(-14, -1) < DEVSMOOTHED(-3, 3)) << 6) + ((DEVSMOOTHED(1, -2) < DEVSMOOTHED(11, 2)) << 5) + ((DEVSMOOTHED(1, 6) < DEVSMOOTHED(-10, -7)) << 4) + ((DEVSMOOTHED(13, 2) < DEVSMOOTHED(-1, 0)) << 3) + ((DEVSMOOTHED(-14, 5) < DEVSMOOTHED(5, -3)) << 2) + ((DEVSMOOTHED(-2, 8) < DEVSMOOTHED(2, 4)) << 1) + ((DEVSMOOTHED(-11, 8) < DEVSMOOTHED(-15, 5)) << 0));
	desc[1] = (uchar)(((DEVSMOOTHED(-6, -23) < DEVSMOOTHED(8, -9)) << 7) + ((DEVSMOOTHED(-12, 6) < DEVSMOOTHED(-10, 8)) << 6) + ((DEVSMOOTHED(-3, -1) < DEVSMOOTHED(8, 1)) << 5) + ((DEVSMOOTHED(3, 6) < DEVSMOOTHED(5, 6)) << 4) + ((DEVSMOOTHED(-7, -6) < DEVSMOOTHED(5, -5)) << 3) + ((DEVSMOOTHED(22, -2) < DEVSMOOTHED(-11, -8)) << 2) + ((DEVSMOOTHED(14, 7) < DEVSMOOTHED(8, 5)) << 1) + ((DEVSMOOTHED(-1, 14) < DEVSMOOTHED(-5, -14)) << 0));
	desc[2] = (uchar)(((DEVSMOOTHED(-14, 9) < DEVSMOOTHED(2, 0)) << 7) + ((DEVSMOOTHED(7, -3) < DEVSMOOTHED(22, 6)) << 6) + ((DEVSMOOTHED(-6, 6) < DEVSMOOTHED(-8, -5)) << 5) + ((DEVSMOOTHED(-5, 9) < DEVSMOOTHED(7, -1)) << 4) + ((DEVSMOOTHED(-3, -7) < DEVSMOOTHED(-10, -18)) << 3) + ((DEVSMOOTHED(4, -5) < DEVSMOOTHED(0, 11)) << 2) + ((DEVSMOOTHED(2, 3) < DEVSMOOTHED(9, 10)) << 1) + ((DEVSMOOTHED(-10, 3) < DEVSMOOTHED(4, 9)) << 0));
	desc[3] = (uchar)(((DEVSMOOTHED(0, 12) < DEVSMOOTHED(-3, 19)) << 7) + ((DEVSMOOTHED(1, 15) < DEVSMOOTHED(-11, -5)) << 6) + ((DEVSMOOTHED(14, -1) < DEVSMOOTHED(7, 8)) << 5) + ((DEVSMOOTHED(7, -23) < DEVSMOOTHED(-5, 5)) << 4) + ((DEVSMOOTHED(0, -6) < DEVSMOOTHED(-10, 17)) << 3) + ((DEVSMOOTHED(13, -4) < DEVSMOOTHED(-3, -4)) << 2) + ((DEVSMOOTHED(-12, 1) < DEVSMOOTHED(-12, 2)) << 1) + ((DEVSMOOTHED(0, 8) < DEVSMOOTHED(3, 22)) << 0));
	desc[4] = (uchar)(((DEVSMOOTHED(-13, 13) < DEVSMOOTHED(3, -1)) << 7) + ((DEVSMOOTHED(-16, 17) < DEVSMOOTHED(6, 10)) << 6) + ((DEVSMOOTHED(7, 15) < DEVSMOOTHED(-5, 0)) << 5) + ((DEVSMOOTHED(2, -12) < DEVSMOOTHED(19, -2)) << 4) + ((DEVSMOOTHED(3, -6) < DEVSMOOTHED(-4, -15)) << 3) + ((DEVSMOOTHED(8, 3) < DEVSMOOTHED(0, 14)) << 2) + ((DEVSMOOTHED(4, -11) < DEVSMOOTHED(5, 5)) << 1) + ((DEVSMOOTHED(11, -7) < DEVSMOOTHED(7, 1)) << 0));
	desc[5] = (uchar)(((DEVSMOOTHED(6, 12) < DEVSMOOTHED(21, 3)) << 7) + ((DEVSMOOTHED(-3, 2) < DEVSMOOTHED(14, 1)) << 6) + ((DEVSMOOTHED(5, 1) < DEVSMOOTHED(-5, 11)) << 5) + ((DEVSMOOTHED(3, -17) < DEVSMOOTHED(-6, 2)) << 4) + ((DEVSMOOTHED(6, 8) < DEVSMOOTHED(5, -10)) << 3) + ((DEVSMOOTHED(-14, -2) < DEVSMOOTHED(0, 4)) << 2) + ((DEVSMOOTHED(5, -7) < DEVSMOOTHED(-6, 5)) << 1) + ((DEVSMOOTHED(10, 4) < DEVSMOOTHED(4, -7)) << 0));
	desc[6] = (uchar)(((DEVSMOOTHED(22, 0) < DEVSMOOTHED(7, -18)) << 7) + ((DEVSMOOTHED(-1, -3) < DEVSMOOTHED(0, 18)) << 6) + ((DEVSMOOTHED(-4, 22) < DEVSMOOTHED(-5, 3)) << 5) + ((DEVSMOOTHED(1, -7) < DEVSMOOTHED(2, -3)) << 4) + ((DEVSMOOTHED(19, -20) < DEVSMOOTHED(17, -2)) << 3) + ((DEVSMOOTHED(3, -10) < DEVSMOOTHED(-8, 24)) << 2) + ((DEVSMOOTHED(-5, -14) < DEVSMOOTHED(7, 5)) << 1) + ((DEVSMOOTHED(-2, 12) < DEVSMOOTHED(-4, -15)) << 0));
	desc[7] = (uchar)(((DEVSMOOTHED(4, 12) < DEVSMOOTHED(0, -19)) << 7) + ((DEVSMOOTHED(20, 13) < DEVSMOOTHED(3, 5)) << 6) + ((DEVSMOOTHED(-8, -12) < DEVSMOOTHED(5, 0)) << 5) + ((DEVSMOOTHED(-5, 6) < DEVSMOOTHED(-7, -11)) << 4) + ((DEVSMOOTHED(6, -11) < DEVSMOOTHED(-3, -22)) << 3) + ((DEVSMOOTHED(15, 4) < DEVSMOOTHED(10, 1)) << 2) + ((DEVSMOOTHED(-7, -4) < DEVSMOOTHED(15, -6)) << 1) + ((DEVSMOOTHED(5, 10) < DEVSMOOTHED(0, 24)) << 0));
	desc[8] = (uchar)(((DEVSMOOTHED(3, 6) < DEVSMOOTHED(22, -2)) << 7) + ((DEVSMOOTHED(-13, 14) < DEVSMOOTHED(4, -4)) << 6) + ((DEVSMOOTHED(-13, 8) < DEVSMOOTHED(-18, -22)) << 5) + ((DEVSMOOTHED(-1, -1) < DEVSMOOTHED(-7, 3)) << 4) + ((DEVSMOOTHED(-19, -12) < DEVSMOOTHED(4, 3)) << 3) + ((DEVSMOOTHED(8, 10) < DEVSMOOTHED(13, -2)) << 2) + ((DEVSMOOTHED(-6, -1) < DEVSMOOTHED(-6, -5)) << 1) + ((DEVSMOOTHED(2, -21) < DEVSMOOTHED(-3, 2)) << 0));
	desc[9] = (uchar)(((DEVSMOOTHED(4, -7) < DEVSMOOTHED(0, 16)) << 7) + ((DEVSMOOTHED(-6, -5) < DEVSMOOTHED(-12, -1)) << 6) + ((DEVSMOOTHED(1, -1) < DEVSMOOTHED(9, 18)) << 5) + ((DEVSMOOTHED(-7, 10) < DEVSMOOTHED(-11, 6)) << 4) + ((DEVSMOOTHED(4, 3) < DEVSMOOTHED(19, -7)) << 3) + ((DEVSMOOTHED(-18, 5) < DEVSMOOTHED(-4, 5)) << 2) + ((DEVSMOOTHED(4, 0) < DEVSMOOTHED(-20, 4)) << 1) + ((DEVSMOOTHED(7, -11) < DEVSMOOTHED(18, 12)) << 0));
	desc[10] = (uchar)(((DEVSMOOTHED(-20, 17) < DEVSMOOTHED(-18, 7)) << 7) + ((DEVSMOOTHED(2, 15) < DEVSMOOTHED(19, -11)) << 6) + ((DEVSMOOTHED(-18, 6) < DEVSMOOTHED(-7, 3)) << 5) + ((DEVSMOOTHED(-4, 1) < DEVSMOOTHED(-14, 13)) << 4) + ((DEVSMOOTHED(17, 3) < DEVSMOOTHED(2, -8)) << 3) + ((DEVSMOOTHED(-7, 2) < DEVSMOOTHED(1, 6)) << 2) + ((DEVSMOOTHED(17, -9) < DEVSMOOTHED(-2, 8)) << 1) + ((DEVSMOOTHED(-8, -6) < DEVSMOOTHED(-1, 12)) << 0));
	desc[11] = (uchar)(((DEVSMOOTHED(-2, 4) < DEVSMOOTHED(-1, 6)) << 7) + ((DEVSMOOTHED(-2, 7) < DEVSMOOTHED(6, 8)) << 6) + ((DEVSMOOTHED(-8, -1) < DEVSMOOTHED(-7, -9)) << 5) + ((DEVSMOOTHED(8, -9) < DEVSMOOTHED(15, 0)) << 4) + ((DEVSMOOTHED(0, 22) < DEVSMOOTHED(-4, -15)) << 3) + ((DEVSMOOTHED(-14, -1) < DEVSMOOTHED(3, -2)) << 2) + ((DEVSMOOTHED(-7, -4) < DEVSMOOTHED(17, -7)) << 1) + ((DEVSMOOTHED(-8, -2) < DEVSMOOTHED(9, -4)) << 0));
	desc[12] = (uchar)(((DEVSMOOTHED(5, -7) < DEVSMOOTHED(7, 7)) << 7) + ((DEVSMOOTHED(-5, 13) < DEVSMOOTHED(-8, 11)) << 6) + ((DEVSMOOTHED(11, -4) < DEVSMOOTHED(0, 8)) << 5) + ((DEVSMOOTHED(5, -11) < DEVSMOOTHED(-9, -6)) << 4) + ((DEVSMOOTHED(2, -6) < DEVSMOOTHED(3, -20)) << 3) + ((DEVSMOOTHED(-6, 2) < DEVSMOOTHED(6, 10)) << 2) + ((DEVSMOOTHED(-6, -6) < DEVSMOOTHED(-15, 7)) << 1) + ((DEVSMOOTHED(-6, -3) < DEVSMOOTHED(2, 1)) << 0));
	desc[13] = (uchar)(((DEVSMOOTHED(11, 0) < DEVSMOOTHED(-3, 2)) << 7) + ((DEVSMOOTHED(7, -12) < DEVSMOOTHED(14, 5)) << 6) + ((DEVSMOOTHED(0, -7) < DEVSMOOTHED(-1, -1)) << 5) + ((DEVSMOOTHED(-16, 0) < DEVSMOOTHED(6, 8)) << 4) + ((DEVSMOOTHED(22, 11) < DEVSMOOTHED(0, -3)) << 3) + ((DEVSMOOTHED(19, 0) < DEVSMOOTHED(5, -17)) << 2) + ((DEVSMOOTHED(-23, -14) < DEVSMOOTHED(-13, -19)) << 1) + ((DEVSMOOTHED(-8, 10) < DEVSMOOTHED(-11, -2)) << 0));
	desc[14] = (uchar)(((DEVSMOOTHED(-11, 6) < DEVSMOOTHED(-10, 13)) << 7) + ((DEVSMOOTHED(1, -7) < DEVSMOOTHED(14, 0)) << 6) + ((DEVSMOOTHED(-12, 1) < DEVSMOOTHED(-5, -5)) << 5) + ((DEVSMOOTHED(4, 7) < DEVSMOOTHED(8, -1)) << 4) + ((DEVSMOOTHED(-1, -5) < DEVSMOOTHED(15, 2)) << 3) + ((DEVSMOOTHED(-3, -1) < DEVSMOOTHED(7, -10)) << 2) + ((DEVSMOOTHED(3, -6) < DEVSMOOTHED(10, -18)) << 1) + ((DEVSMOOTHED(-7, -13) < DEVSMOOTHED(-13, 10)) << 0));
	desc[15] = (uchar)(((DEVSMOOTHED(1, -1) < DEVSMOOTHED(13, -10)) << 7) + ((DEVSMOOTHED(-19, 14) < DEVSMOOTHED(8, -14)) << 6) + ((DEVSMOOTHED(-4, -13) < DEVSMOOTHED(7, 1)) << 5) + ((DEVSMOOTHED(1, -2) < DEVSMOOTHED(12, -7)) << 4) + ((DEVSMOOTHED(3, -5) < DEVSMOOTHED(1, -5)) << 3) + ((DEVSMOOTHED(-2, -2) < DEVSMOOTHED(8, -10)) << 2) + ((DEVSMOOTHED(2, 14) < DEVSMOOTHED(8, 7)) << 1) + ((DEVSMOOTHED(3, 9) < DEVSMOOTHED(8, 2)) << 0));
	desc[16] = (uchar)(((DEVSMOOTHED(-9, 1) < DEVSMOOTHED(-18, 0)) << 7) + ((DEVSMOOTHED(4, 0) < DEVSMOOTHED(1, 12)) << 6) + ((DEVSMOOTHED(0, 9) < DEVSMOOTHED(-14, -10)) << 5) + ((DEVSMOOTHED(-13, -9) < DEVSMOOTHED(-2, 6)) << 4) + ((DEVSMOOTHED(1, 5) < DEVSMOOTHED(10, 10)) << 3) + ((DEVSMOOTHED(-3, -6) < DEVSMOOTHED(-16, -5)) << 2) + ((DEVSMOOTHED(11, 6) < DEVSMOOTHED(-5, 0)) << 1) + ((DEVSMOOTHED(-23, 10) < DEVSMOOTHED(1, 2)) << 0));
	desc[17] = (uchar)(((DEVSMOOTHED(13, -5) < DEVSMOOTHED(-3, 9)) << 7) + ((DEVSMOOTHED(-4, -1) < DEVSMOOTHED(-13, -5)) << 6) + ((DEVSMOOTHED(10, 13) < DEVSMOOTHED(-11, 8)) << 5) + ((DEVSMOOTHED(19, 20) < DEVSMOOTHED(-9, 2)) << 4) + ((DEVSMOOTHED(4, -8) < DEVSMOOTHED(0, -9)) << 3) + ((DEVSMOOTHED(-14, 10) < DEVSMOOTHED(15, 19)) << 2) + ((DEVSMOOTHED(-14, -12) < DEVSMOOTHED(-10, -3)) << 1) + ((DEVSMOOTHED(-23, -3) < DEVSMOOTHED(17, -2)) << 0));
	desc[18] = (uchar)(((DEVSMOOTHED(-3, -11) < DEVSMOOTHED(6, -14)) << 7) + ((DEVSMOOTHED(19, -2) < DEVSMOOTHED(-4, 2)) << 6) + ((DEVSMOOTHED(-5, 5) < DEVSMOOTHED(3, -13)) << 5) + ((DEVSMOOTHED(2, -2) < DEVSMOOTHED(-5, 4)) << 4) + ((DEVSMOOTHED(17, 4) < DEVSMOOTHED(17, -11)) << 3) + ((DEVSMOOTHED(-7, -2) < DEVSMOOTHED(1, 23)) << 2) + ((DEVSMOOTHED(8, 13) < DEVSMOOTHED(1, -16)) << 1) + ((DEVSMOOTHED(-13, -5) < DEVSMOOTHED(1, -17)) << 0));
	desc[19] = (uchar)(((DEVSMOOTHED(4, 6) < DEVSMOOTHED(-8, -3)) << 7) + ((DEVSMOOTHED(-5, -9) < DEVSMOOTHED(-2, -10)) << 6) + ((DEVSMOOTHED(-9, 0) < DEVSMOOTHED(-7, -2)) << 5) + ((DEVSMOOTHED(5, 0) < DEVSMOOTHED(5, 2)) << 4) + ((DEVSMOOTHED(-4, -16) < DEVSMOOTHED(6, 3)) << 3) + ((DEVSMOOTHED(2, -15) < DEVSMOOTHED(-2, 12)) << 2) + ((DEVSMOOTHED(4, -1) < DEVSMOOTHED(6, 2)) << 1) + ((DEVSMOOTHED(1, 1) < DEVSMOOTHED(-2, -8)) << 0));
	desc[20] = (uchar)(((DEVSMOOTHED(-2, 12) < DEVSMOOTHED(-5, -2)) << 7) + ((DEVSMOOTHED(-8, 8) < DEVSMOOTHED(-9, 9)) << 6) + ((DEVSMOOTHED(2, -10) < DEVSMOOTHED(3, 1)) << 5) + ((DEVSMOOTHED(-4, 10) < DEVSMOOTHED(-9, 4)) << 4) + ((DEVSMOOTHED(6, 12) < DEVSMOOTHED(2, 5)) << 3) + ((DEVSMOOTHED(-3, -8) < DEVSMOOTHED(0, 5)) << 2) + ((DEVSMOOTHED(-13, 1) < DEVSMOOTHED(-7, 2)) << 1) + ((DEVSMOOTHED(-1, -10) < DEVSMOOTHED(7, -18)) << 0));
	desc[21] = (uchar)(((DEVSMOOTHED(-1, 8) < DEVSMOOTHED(-9, -10)) << 7) + ((DEVSMOOTHED(-23, -1) < DEVSMOOTHED(6, 2)) << 6) + ((DEVSMOOTHED(-5, -3) < DEVSMOOTHED(3, 2)) << 5) + ((DEVSMOOTHED(0, 11) < DEVSMOOTHED(-4, -7)) << 4) + ((DEVSMOOTHED(15, 2) < DEVSMOOTHED(-10, -3)) << 3) + ((DEVSMOOTHED(-20, -8) < DEVSMOOTHED(-13, 3)) << 2) + ((DEVSMOOTHED(-19, -12) < DEVSMOOTHED(5, -11)) << 1) + ((DEVSMOOTHED(-17, -13) < DEVSMOOTHED(-3, 2)) << 0));
	desc[22] = (uchar)(((DEVSMOOTHED(7, 4) < DEVSMOOTHED(-12, 0)) << 7) + ((DEVSMOOTHED(5, -1) < DEVSMOOTHED(-14, -6)) << 6) + ((DEVSMOOTHED(-4, 11) < DEVSMOOTHED(0, -4)) << 5) + ((DEVSMOOTHED(3, 10) < DEVSMOOTHED(7, -3)) << 4) + ((DEVSMOOTHED(13, 21) < DEVSMOOTHED(-11, 6)) << 3) + ((DEVSMOOTHED(-12, 24) < DEVSMOOTHED(-7, -4)) << 2) + ((DEVSMOOTHED(4, 16) < DEVSMOOTHED(3, -14)) << 1) + ((DEVSMOOTHED(-3, 5) < DEVSMOOTHED(-7, -12)) << 0));
	desc[23] = (uchar)(((DEVSMOOTHED(0, -4) < DEVSMOOTHED(7, -5)) << 7) + ((DEVSMOOTHED(-17, -9) < DEVSMOOTHED(13, -7)) << 6) + ((DEVSMOOTHED(22, -6) < DEVSMOOTHED(-11, 5)) << 5) + ((DEVSMOOTHED(2, -8) < DEVSMOOTHED(23, -11)) << 4) + ((DEVSMOOTHED(7, -10) < DEVSMOOTHED(-1, 14)) << 3) + ((DEVSMOOTHED(-3, -10) < DEVSMOOTHED(8, 3)) << 2) + ((DEVSMOOTHED(-13, 1) < DEVSMOOTHED(-6, 0)) << 1) + ((DEVSMOOTHED(-7, -21) < DEVSMOOTHED(6, -14)) << 0));
	desc[24] = (uchar)(((DEVSMOOTHED(18, 19) < DEVSMOOTHED(-4, -6)) << 7) + ((DEVSMOOTHED(10, 7) < DEVSMOOTHED(-1, -4)) << 6) + ((DEVSMOOTHED(-1, 21) < DEVSMOOTHED(1, -5)) << 5) + ((DEVSMOOTHED(-10, 6) < DEVSMOOTHED(-11, -2)) << 4) + ((DEVSMOOTHED(18, -3) < DEVSMOOTHED(-1, 7)) << 3) + ((DEVSMOOTHED(-3, -9) < DEVSMOOTHED(-5, 10)) << 2) + ((DEVSMOOTHED(-13, 14) < DEVSMOOTHED(17, -3)) << 1) + ((DEVSMOOTHED(11, -19) < DEVSMOOTHED(-1, -18)) << 0));
	desc[25] = (uchar)(((DEVSMOOTHED(8, -2) < DEVSMOOTHED(-18, -23)) << 7) + ((DEVSMOOTHED(0, -5) < DEVSMOOTHED(-2, -9)) << 6) + ((DEVSMOOTHED(-4, -11) < DEVSMOOTHED(2, -8)) << 5) + ((DEVSMOOTHED(14, 6) < DEVSMOOTHED(-3, -6)) << 4) + ((DEVSMOOTHED(-3, 0) < DEVSMOOTHED(-15, 0)) << 3) + ((DEVSMOOTHED(-9, 4) < DEVSMOOTHED(-15, -9)) << 2) + ((DEVSMOOTHED(-1, 11) < DEVSMOOTHED(3, 11)) << 1) + ((DEVSMOOTHED(-10, -16) < DEVSMOOTHED(-7, 7)) << 0));
	desc[26] = (uchar)(((DEVSMOOTHED(-2, -10) < DEVSMOOTHED(-10, -2)) << 7) + ((DEVSMOOTHED(-5, -3) < DEVSMOOTHED(5, -23)) << 6) + ((DEVSMOOTHED(13, -8) < DEVSMOOTHED(-15, -11)) << 5) + ((DEVSMOOTHED(-15, 11) < DEVSMOOTHED(6, -6)) << 4) + ((DEVSMOOTHED(-16, -3) < DEVSMOOTHED(-2, 2)) << 3) + ((DEVSMOOTHED(6, 12) < DEVSMOOTHED(-16, 24)) << 2) + ((DEVSMOOTHED(-10, 0) < DEVSMOOTHED(8, 11)) << 1) + ((DEVSMOOTHED(-7, 7) < DEVSMOOTHED(-19, -7)) << 0));
	desc[27] = (uchar)(((DEVSMOOTHED(5, 16) < DEVSMOOTHED(9, -3)) << 7) + ((DEVSMOOTHED(9, 7) < DEVSMOOTHED(-7, -16)) << 6) + ((DEVSMOOTHED(3, 2) < DEVSMOOTHED(-10, 9)) << 5) + ((DEVSMOOTHED(21, 1) < DEVSMOOTHED(8, 7)) << 4) + ((DEVSMOOTHED(7, 0) < DEVSMOOTHED(1, 17)) << 3) + ((DEVSMOOTHED(-8, 12) < DEVSMOOTHED(9, 6)) << 2) + ((DEVSMOOTHED(11, -7) < DEVSMOOTHED(-8, -6)) << 1) + ((DEVSMOOTHED(19, 0) < DEVSMOOTHED(9, 3)) << 0));
	desc[28] = (uchar)(((DEVSMOOTHED(1, -7) < DEVSMOOTHED(-5, -11)) << 7) + ((DEVSMOOTHED(0, 8) < DEVSMOOTHED(-2, 14)) << 6) + ((DEVSMOOTHED(12, -2) < DEVSMOOTHED(-15, -6)) << 5) + ((DEVSMOOTHED(4, 12) < DEVSMOOTHED(0, -21)) << 4) + ((DEVSMOOTHED(17, -4) < DEVSMOOTHED(-6, -7)) << 3) + ((DEVSMOOTHED(-10, -9) < DEVSMOOTHED(-14, -7)) << 2) + ((DEVSMOOTHED(-15, -10) < DEVSMOOTHED(-15, -14)) << 1) + ((DEVSMOOTHED(-7, -5) < DEVSMOOTHED(5, -12)) << 0));
	desc[29] = (uchar)(((DEVSMOOTHED(-4, 0) < DEVSMOOTHED(15, -4)) << 7) + ((DEVSMOOTHED(5, 2) < DEVSMOOTHED(-6, -23)) << 6) + ((DEVSMOOTHED(-4, -21) < DEVSMOOTHED(-6, 4)) << 5) + ((DEVSMOOTHED(-10, 5) < DEVSMOOTHED(-15, 6)) << 4) + ((DEVSMOOTHED(4, -3) < DEVSMOOTHED(-1, 5)) << 3) + ((DEVSMOOTHED(-4, 19) < DEVSMOOTHED(-23, -4)) << 2) + ((DEVSMOOTHED(-4, 17) < DEVSMOOTHED(13, -11)) << 1) + ((DEVSMOOTHED(1, 12) < DEVSMOOTHED(4, -14)) << 0));
	desc[30] = (uchar)(((DEVSMOOTHED(-11, -6) < DEVSMOOTHED(-20, 10)) << 7) + ((DEVSMOOTHED(4, 5) < DEVSMOOTHED(3, 20)) << 6) + ((DEVSMOOTHED(-8, -20) < DEVSMOOTHED(3, 1)) << 5) + ((DEVSMOOTHED(-19, 9) < DEVSMOOTHED(9, -3)) << 4) + ((DEVSMOOTHED(18, 15) < DEVSMOOTHED(11, -4)) << 3) + ((DEVSMOOTHED(12, 16) < DEVSMOOTHED(8, 7)) << 2) + ((DEVSMOOTHED(-14, -8) < DEVSMOOTHED(-3, 9)) << 1) + ((DEVSMOOTHED(-6, 0) < DEVSMOOTHED(2, -4)) << 0));
	desc[31] = (uchar)(((DEVSMOOTHED(1, -10) < DEVSMOOTHED(-1, 2)) << 7) + ((DEVSMOOTHED(8, -7) < DEVSMOOTHED(-6, 18)) << 6) + ((DEVSMOOTHED(9, 12) < DEVSMOOTHED(-7, -23)) << 5) + ((DEVSMOOTHED(8, -6) < DEVSMOOTHED(5, 2)) << 4) + ((DEVSMOOTHED(-9, 6) < DEVSMOOTHED(-12, -7)) << 3) + ((DEVSMOOTHED(-1, -2) < DEVSMOOTHED(-7, 2)) << 2) + ((DEVSMOOTHED(9, 9) < DEVSMOOTHED(7, 15)) << 1) + ((DEVSMOOTHED(6, 2) < DEVSMOOTHED(-6, 6)) << 0));
#undef DEVSMOOTHED

}




cudaError_t MyCudaUtils::ComputeBriefDescriptors(cv::Mat& image, std::vector<cv::KeyPoint>& kps, cv::Mat& desc, int descSize) 
{

	if (descSize != 32) {
		std::cout << "Descriptor sizes other than 32 bytes currently not implemented" << std::endl;
		std::cout << "Press q to exit the program or any other key to continue: ";
		char c;
		std::cin >> c;
		if ('q' == c)
			exit(EXIT_FAILURE);
	}
		

	// Convert to greyscale if required
	cv::Mat grayImage = image;
	if (image.type() != CV_8U) cv::cvtColor(image, grayImage, CV_BGR2GRAY);

	// Compute the integral image for smoothing
	cv::Mat intImage;
	cv::integral(grayImage, intImage, CV_32S);

	//Remove keypoints very close to the border
	static const int PATCH_SIZE = 48;											// Size of patch used to compute descriptors - 24 x 24 pixel window
	static const int KERNEL_SIZE = 9;											// Size of filtering Kernel used on each descriptor point to compare 9x9 pixel window
	runByImageBorder(kps, image.size(), PATCH_SIZE / 2 + KERNEL_SIZE / 2);		// We don't want our patch or kernel to overflow to the edge so offset by both of them

	// Initialise list of descriptors to zero
	desc = cv::Mat::zeros((int)kps.size(), descSize, CV_8U);



	int knp = 2;		// Number of params describing a keypoint
	int imSize = intImage.rows * intImage.cols;


	// Allocate memory
	int* dev_intImage;
	float* dev_kps;
	unsigned char* dev_desc;
	try {
		CUDA_CHECK(cudaSetDevice(0)); // I dunno about this because it means you can't see the error type involved. 

		CUDA_CHECK(cudaMalloc((void**)&dev_intImage, imSize * sizeof(int)));

		CUDA_CHECK(cudaMalloc((void**)&dev_kps, kps.size() * knp * sizeof(float)));

		CUDA_CHECK(cudaMalloc((void**)&dev_desc, kps.size() * descSize * sizeof(unsigned char)));

		// Copy the integral image and initialise descriptors to zero
		CUDA_CHECK(cudaMemcpy(dev_intImage, intImage.data, imSize * sizeof(int), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(dev_desc, desc.data, kps.size() * descSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

		// Copy the keypoints into an array and then onto the device
		int num_kps = kps.size();
		float* kpsArray = new float[num_kps * knp];   // SQUARE BRACKETS FOR DYNAMIC MEMORY 
		float* kpsXArray = kpsArray;
		float* kpsYArray = kpsArray + num_kps;
		for (int i = 0; i < num_kps; i++) {
			kpsXArray[i] = kps[i].pt.x;
			kpsYArray[i] = kps[i].pt.y;
		}
		CUDA_CHECK(cudaMemcpy(dev_kps, kpsArray, num_kps * knp * sizeof(float), cudaMemcpyHostToDevice));
		delete[] kpsArray;

		// Launch the Kernel
		int blocks = cvCeil(num_kps / 128);
		pixelTest32Kernel << < blocks, 128 >> >(dev_intImage, intImage.cols, dev_kps, num_kps, dev_desc);

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		// Download the output
		CUDA_CHECK(cudaMemcpy(desc.data, dev_desc, kps.size() * descSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		throw (cudaSuccess);

	}
	catch (cudaError_t cudaStatus) {
		cudaFree(dev_intImage);
		cudaFree(dev_kps);
		cudaFree(dev_desc);

		return cudaStatus;
	}


}













/**************************************************************************************
*									SERIAL BRIEF
**************************************************************************************/
//// This computes the boxed filter smoothing of the image (which is a just an average
//// and so can be computed by addition and subtraction of appropriate corners in the
//// integral image)
//inline int smoothedSum(const cv::Mat& sum, const cv::KeyPoint& pt, int y, int x)
//{
//	static const int KERNEL_SIZE = 9;
//	static const int HALF_KERNEL = KERNEL_SIZE / 2;
//
//	int img_y = (int)(pt.pt.y + 0.5) + y;
//	int img_x = (int)(pt.pt.x + 0.5) + x;	// Add 0.5 and cast to int automatically rounds up I believe assuming pt.pt.x is float (keypoints can have sub-pixel precision
//	return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)		// The four corners of the integral image
//		- sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
//		- sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
//		+ sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
//}



//// Computes the response with a 32byte descriptor of all the keypoints in the set
//// It iterates through all the keypoints, and then defines the SMOOTHED function
//// to be called with the given integral image and the current keypoint coordinates. 
//static void pixelTests32(const cv::Mat& sum, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
//{
//	// So this would be nicely parallelisable, although I'm not sure how CUDA likes #defines, or whether we could even include the function in there. 
//	// Okay so we only need to mark smoothedSum as GPU function using the __device__ specifier to make sure that it can be accessed inside the Kernel
//	// Anyway the point is we can have each thread doing one of these just fine since they are all identical - and all the other stuff is so much fluffing
//	// around. 
//	for (int i = 0; i < (int)keypoints.size(); ++i)
//	{
//		uchar* desc = descriptors.ptr(i);
//		const cv::KeyPoint& pt = keypoints[i];
//
//#define SMOOTHED(y,x) smoothedSum(sum, pt, y, x)
//		// Eight binary comparisons contribute a single bit in the descriptor each byte, bit shifted into place in 1 of the 32 total bytes 
//		desc[0] = (uchar)(((SMOOTHED(-2, -1) < SMOOTHED(7, -1)) << 7) + ((SMOOTHED(-14, -1) < SMOOTHED(-3, 3)) << 6) + ((SMOOTHED(1, -2) < SMOOTHED(11, 2)) << 5) + ((SMOOTHED(1, 6) < SMOOTHED(-10, -7)) << 4) + ((SMOOTHED(13, 2) < SMOOTHED(-1, 0)) << 3) + ((SMOOTHED(-14, 5) < SMOOTHED(5, -3)) << 2) + ((SMOOTHED(-2, 8) < SMOOTHED(2, 4)) << 1) + ((SMOOTHED(-11, 8) < SMOOTHED(-15, 5)) << 0));
//		desc[1] = (uchar)(((SMOOTHED(-6, -23) < SMOOTHED(8, -9)) << 7) + ((SMOOTHED(-12, 6) < SMOOTHED(-10, 8)) << 6) + ((SMOOTHED(-3, -1) < SMOOTHED(8, 1)) << 5) + ((SMOOTHED(3, 6) < SMOOTHED(5, 6)) << 4) + ((SMOOTHED(-7, -6) < SMOOTHED(5, -5)) << 3) + ((SMOOTHED(22, -2) < SMOOTHED(-11, -8)) << 2) + ((SMOOTHED(14, 7) < SMOOTHED(8, 5)) << 1) + ((SMOOTHED(-1, 14) < SMOOTHED(-5, -14)) << 0));
//		desc[2] = (uchar)(((SMOOTHED(-14, 9) < SMOOTHED(2, 0)) << 7) + ((SMOOTHED(7, -3) < SMOOTHED(22, 6)) << 6) + ((SMOOTHED(-6, 6) < SMOOTHED(-8, -5)) << 5) + ((SMOOTHED(-5, 9) < SMOOTHED(7, -1)) << 4) + ((SMOOTHED(-3, -7) < SMOOTHED(-10, -18)) << 3) + ((SMOOTHED(4, -5) < SMOOTHED(0, 11)) << 2) + ((SMOOTHED(2, 3) < SMOOTHED(9, 10)) << 1) + ((SMOOTHED(-10, 3) < SMOOTHED(4, 9)) << 0));
//		desc[3] = (uchar)(((SMOOTHED(0, 12) < SMOOTHED(-3, 19)) << 7) + ((SMOOTHED(1, 15) < SMOOTHED(-11, -5)) << 6) + ((SMOOTHED(14, -1) < SMOOTHED(7, 8)) << 5) + ((SMOOTHED(7, -23) < SMOOTHED(-5, 5)) << 4) + ((SMOOTHED(0, -6) < SMOOTHED(-10, 17)) << 3) + ((SMOOTHED(13, -4) < SMOOTHED(-3, -4)) << 2) + ((SMOOTHED(-12, 1) < SMOOTHED(-12, 2)) << 1) + ((SMOOTHED(0, 8) < SMOOTHED(3, 22)) << 0));
//		desc[4] = (uchar)(((SMOOTHED(-13, 13) < SMOOTHED(3, -1)) << 7) + ((SMOOTHED(-16, 17) < SMOOTHED(6, 10)) << 6) + ((SMOOTHED(7, 15) < SMOOTHED(-5, 0)) << 5) + ((SMOOTHED(2, -12) < SMOOTHED(19, -2)) << 4) + ((SMOOTHED(3, -6) < SMOOTHED(-4, -15)) << 3) + ((SMOOTHED(8, 3) < SMOOTHED(0, 14)) << 2) + ((SMOOTHED(4, -11) < SMOOTHED(5, 5)) << 1) + ((SMOOTHED(11, -7) < SMOOTHED(7, 1)) << 0));
//		desc[5] = (uchar)(((SMOOTHED(6, 12) < SMOOTHED(21, 3)) << 7) + ((SMOOTHED(-3, 2) < SMOOTHED(14, 1)) << 6) + ((SMOOTHED(5, 1) < SMOOTHED(-5, 11)) << 5) + ((SMOOTHED(3, -17) < SMOOTHED(-6, 2)) << 4) + ((SMOOTHED(6, 8) < SMOOTHED(5, -10)) << 3) + ((SMOOTHED(-14, -2) < SMOOTHED(0, 4)) << 2) + ((SMOOTHED(5, -7) < SMOOTHED(-6, 5)) << 1) + ((SMOOTHED(10, 4) < SMOOTHED(4, -7)) << 0));
//		desc[6] = (uchar)(((SMOOTHED(22, 0) < SMOOTHED(7, -18)) << 7) + ((SMOOTHED(-1, -3) < SMOOTHED(0, 18)) << 6) + ((SMOOTHED(-4, 22) < SMOOTHED(-5, 3)) << 5) + ((SMOOTHED(1, -7) < SMOOTHED(2, -3)) << 4) + ((SMOOTHED(19, -20) < SMOOTHED(17, -2)) << 3) + ((SMOOTHED(3, -10) < SMOOTHED(-8, 24)) << 2) + ((SMOOTHED(-5, -14) < SMOOTHED(7, 5)) << 1) + ((SMOOTHED(-2, 12) < SMOOTHED(-4, -15)) << 0));
//		desc[7] = (uchar)(((SMOOTHED(4, 12) < SMOOTHED(0, -19)) << 7) + ((SMOOTHED(20, 13) < SMOOTHED(3, 5)) << 6) + ((SMOOTHED(-8, -12) < SMOOTHED(5, 0)) << 5) + ((SMOOTHED(-5, 6) < SMOOTHED(-7, -11)) << 4) + ((SMOOTHED(6, -11) < SMOOTHED(-3, -22)) << 3) + ((SMOOTHED(15, 4) < SMOOTHED(10, 1)) << 2) + ((SMOOTHED(-7, -4) < SMOOTHED(15, -6)) << 1) + ((SMOOTHED(5, 10) < SMOOTHED(0, 24)) << 0));
//		desc[8] = (uchar)(((SMOOTHED(3, 6) < SMOOTHED(22, -2)) << 7) + ((SMOOTHED(-13, 14) < SMOOTHED(4, -4)) << 6) + ((SMOOTHED(-13, 8) < SMOOTHED(-18, -22)) << 5) + ((SMOOTHED(-1, -1) < SMOOTHED(-7, 3)) << 4) + ((SMOOTHED(-19, -12) < SMOOTHED(4, 3)) << 3) + ((SMOOTHED(8, 10) < SMOOTHED(13, -2)) << 2) + ((SMOOTHED(-6, -1) < SMOOTHED(-6, -5)) << 1) + ((SMOOTHED(2, -21) < SMOOTHED(-3, 2)) << 0));
//		desc[9] = (uchar)(((SMOOTHED(4, -7) < SMOOTHED(0, 16)) << 7) + ((SMOOTHED(-6, -5) < SMOOTHED(-12, -1)) << 6) + ((SMOOTHED(1, -1) < SMOOTHED(9, 18)) << 5) + ((SMOOTHED(-7, 10) < SMOOTHED(-11, 6)) << 4) + ((SMOOTHED(4, 3) < SMOOTHED(19, -7)) << 3) + ((SMOOTHED(-18, 5) < SMOOTHED(-4, 5)) << 2) + ((SMOOTHED(4, 0) < SMOOTHED(-20, 4)) << 1) + ((SMOOTHED(7, -11) < SMOOTHED(18, 12)) << 0));
//		desc[10] = (uchar)(((SMOOTHED(-20, 17) < SMOOTHED(-18, 7)) << 7) + ((SMOOTHED(2, 15) < SMOOTHED(19, -11)) << 6) + ((SMOOTHED(-18, 6) < SMOOTHED(-7, 3)) << 5) + ((SMOOTHED(-4, 1) < SMOOTHED(-14, 13)) << 4) + ((SMOOTHED(17, 3) < SMOOTHED(2, -8)) << 3) + ((SMOOTHED(-7, 2) < SMOOTHED(1, 6)) << 2) + ((SMOOTHED(17, -9) < SMOOTHED(-2, 8)) << 1) + ((SMOOTHED(-8, -6) < SMOOTHED(-1, 12)) << 0));
//		desc[11] = (uchar)(((SMOOTHED(-2, 4) < SMOOTHED(-1, 6)) << 7) + ((SMOOTHED(-2, 7) < SMOOTHED(6, 8)) << 6) + ((SMOOTHED(-8, -1) < SMOOTHED(-7, -9)) << 5) + ((SMOOTHED(8, -9) < SMOOTHED(15, 0)) << 4) + ((SMOOTHED(0, 22) < SMOOTHED(-4, -15)) << 3) + ((SMOOTHED(-14, -1) < SMOOTHED(3, -2)) << 2) + ((SMOOTHED(-7, -4) < SMOOTHED(17, -7)) << 1) + ((SMOOTHED(-8, -2) < SMOOTHED(9, -4)) << 0));
//		desc[12] = (uchar)(((SMOOTHED(5, -7) < SMOOTHED(7, 7)) << 7) + ((SMOOTHED(-5, 13) < SMOOTHED(-8, 11)) << 6) + ((SMOOTHED(11, -4) < SMOOTHED(0, 8)) << 5) + ((SMOOTHED(5, -11) < SMOOTHED(-9, -6)) << 4) + ((SMOOTHED(2, -6) < SMOOTHED(3, -20)) << 3) + ((SMOOTHED(-6, 2) < SMOOTHED(6, 10)) << 2) + ((SMOOTHED(-6, -6) < SMOOTHED(-15, 7)) << 1) + ((SMOOTHED(-6, -3) < SMOOTHED(2, 1)) << 0));
//		desc[13] = (uchar)(((SMOOTHED(11, 0) < SMOOTHED(-3, 2)) << 7) + ((SMOOTHED(7, -12) < SMOOTHED(14, 5)) << 6) + ((SMOOTHED(0, -7) < SMOOTHED(-1, -1)) << 5) + ((SMOOTHED(-16, 0) < SMOOTHED(6, 8)) << 4) + ((SMOOTHED(22, 11) < SMOOTHED(0, -3)) << 3) + ((SMOOTHED(19, 0) < SMOOTHED(5, -17)) << 2) + ((SMOOTHED(-23, -14) < SMOOTHED(-13, -19)) << 1) + ((SMOOTHED(-8, 10) < SMOOTHED(-11, -2)) << 0));
//		desc[14] = (uchar)(((SMOOTHED(-11, 6) < SMOOTHED(-10, 13)) << 7) + ((SMOOTHED(1, -7) < SMOOTHED(14, 0)) << 6) + ((SMOOTHED(-12, 1) < SMOOTHED(-5, -5)) << 5) + ((SMOOTHED(4, 7) < SMOOTHED(8, -1)) << 4) + ((SMOOTHED(-1, -5) < SMOOTHED(15, 2)) << 3) + ((SMOOTHED(-3, -1) < SMOOTHED(7, -10)) << 2) + ((SMOOTHED(3, -6) < SMOOTHED(10, -18)) << 1) + ((SMOOTHED(-7, -13) < SMOOTHED(-13, 10)) << 0));
//		desc[15] = (uchar)(((SMOOTHED(1, -1) < SMOOTHED(13, -10)) << 7) + ((SMOOTHED(-19, 14) < SMOOTHED(8, -14)) << 6) + ((SMOOTHED(-4, -13) < SMOOTHED(7, 1)) << 5) + ((SMOOTHED(1, -2) < SMOOTHED(12, -7)) << 4) + ((SMOOTHED(3, -5) < SMOOTHED(1, -5)) << 3) + ((SMOOTHED(-2, -2) < SMOOTHED(8, -10)) << 2) + ((SMOOTHED(2, 14) < SMOOTHED(8, 7)) << 1) + ((SMOOTHED(3, 9) < SMOOTHED(8, 2)) << 0));
//		desc[16] = (uchar)(((SMOOTHED(-9, 1) < SMOOTHED(-18, 0)) << 7) + ((SMOOTHED(4, 0) < SMOOTHED(1, 12)) << 6) + ((SMOOTHED(0, 9) < SMOOTHED(-14, -10)) << 5) + ((SMOOTHED(-13, -9) < SMOOTHED(-2, 6)) << 4) + ((SMOOTHED(1, 5) < SMOOTHED(10, 10)) << 3) + ((SMOOTHED(-3, -6) < SMOOTHED(-16, -5)) << 2) + ((SMOOTHED(11, 6) < SMOOTHED(-5, 0)) << 1) + ((SMOOTHED(-23, 10) < SMOOTHED(1, 2)) << 0));
//		desc[17] = (uchar)(((SMOOTHED(13, -5) < SMOOTHED(-3, 9)) << 7) + ((SMOOTHED(-4, -1) < SMOOTHED(-13, -5)) << 6) + ((SMOOTHED(10, 13) < SMOOTHED(-11, 8)) << 5) + ((SMOOTHED(19, 20) < SMOOTHED(-9, 2)) << 4) + ((SMOOTHED(4, -8) < SMOOTHED(0, -9)) << 3) + ((SMOOTHED(-14, 10) < SMOOTHED(15, 19)) << 2) + ((SMOOTHED(-14, -12) < SMOOTHED(-10, -3)) << 1) + ((SMOOTHED(-23, -3) < SMOOTHED(17, -2)) << 0));
//		desc[18] = (uchar)(((SMOOTHED(-3, -11) < SMOOTHED(6, -14)) << 7) + ((SMOOTHED(19, -2) < SMOOTHED(-4, 2)) << 6) + ((SMOOTHED(-5, 5) < SMOOTHED(3, -13)) << 5) + ((SMOOTHED(2, -2) < SMOOTHED(-5, 4)) << 4) + ((SMOOTHED(17, 4) < SMOOTHED(17, -11)) << 3) + ((SMOOTHED(-7, -2) < SMOOTHED(1, 23)) << 2) + ((SMOOTHED(8, 13) < SMOOTHED(1, -16)) << 1) + ((SMOOTHED(-13, -5) < SMOOTHED(1, -17)) << 0));
//		desc[19] = (uchar)(((SMOOTHED(4, 6) < SMOOTHED(-8, -3)) << 7) + ((SMOOTHED(-5, -9) < SMOOTHED(-2, -10)) << 6) + ((SMOOTHED(-9, 0) < SMOOTHED(-7, -2)) << 5) + ((SMOOTHED(5, 0) < SMOOTHED(5, 2)) << 4) + ((SMOOTHED(-4, -16) < SMOOTHED(6, 3)) << 3) + ((SMOOTHED(2, -15) < SMOOTHED(-2, 12)) << 2) + ((SMOOTHED(4, -1) < SMOOTHED(6, 2)) << 1) + ((SMOOTHED(1, 1) < SMOOTHED(-2, -8)) << 0));
//		desc[20] = (uchar)(((SMOOTHED(-2, 12) < SMOOTHED(-5, -2)) << 7) + ((SMOOTHED(-8, 8) < SMOOTHED(-9, 9)) << 6) + ((SMOOTHED(2, -10) < SMOOTHED(3, 1)) << 5) + ((SMOOTHED(-4, 10) < SMOOTHED(-9, 4)) << 4) + ((SMOOTHED(6, 12) < SMOOTHED(2, 5)) << 3) + ((SMOOTHED(-3, -8) < SMOOTHED(0, 5)) << 2) + ((SMOOTHED(-13, 1) < SMOOTHED(-7, 2)) << 1) + ((SMOOTHED(-1, -10) < SMOOTHED(7, -18)) << 0));
//		desc[21] = (uchar)(((SMOOTHED(-1, 8) < SMOOTHED(-9, -10)) << 7) + ((SMOOTHED(-23, -1) < SMOOTHED(6, 2)) << 6) + ((SMOOTHED(-5, -3) < SMOOTHED(3, 2)) << 5) + ((SMOOTHED(0, 11) < SMOOTHED(-4, -7)) << 4) + ((SMOOTHED(15, 2) < SMOOTHED(-10, -3)) << 3) + ((SMOOTHED(-20, -8) < SMOOTHED(-13, 3)) << 2) + ((SMOOTHED(-19, -12) < SMOOTHED(5, -11)) << 1) + ((SMOOTHED(-17, -13) < SMOOTHED(-3, 2)) << 0));
//		desc[22] = (uchar)(((SMOOTHED(7, 4) < SMOOTHED(-12, 0)) << 7) + ((SMOOTHED(5, -1) < SMOOTHED(-14, -6)) << 6) + ((SMOOTHED(-4, 11) < SMOOTHED(0, -4)) << 5) + ((SMOOTHED(3, 10) < SMOOTHED(7, -3)) << 4) + ((SMOOTHED(13, 21) < SMOOTHED(-11, 6)) << 3) + ((SMOOTHED(-12, 24) < SMOOTHED(-7, -4)) << 2) + ((SMOOTHED(4, 16) < SMOOTHED(3, -14)) << 1) + ((SMOOTHED(-3, 5) < SMOOTHED(-7, -12)) << 0));
//		desc[23] = (uchar)(((SMOOTHED(0, -4) < SMOOTHED(7, -5)) << 7) + ((SMOOTHED(-17, -9) < SMOOTHED(13, -7)) << 6) + ((SMOOTHED(22, -6) < SMOOTHED(-11, 5)) << 5) + ((SMOOTHED(2, -8) < SMOOTHED(23, -11)) << 4) + ((SMOOTHED(7, -10) < SMOOTHED(-1, 14)) << 3) + ((SMOOTHED(-3, -10) < SMOOTHED(8, 3)) << 2) + ((SMOOTHED(-13, 1) < SMOOTHED(-6, 0)) << 1) + ((SMOOTHED(-7, -21) < SMOOTHED(6, -14)) << 0));
//		desc[24] = (uchar)(((SMOOTHED(18, 19) < SMOOTHED(-4, -6)) << 7) + ((SMOOTHED(10, 7) < SMOOTHED(-1, -4)) << 6) + ((SMOOTHED(-1, 21) < SMOOTHED(1, -5)) << 5) + ((SMOOTHED(-10, 6) < SMOOTHED(-11, -2)) << 4) + ((SMOOTHED(18, -3) < SMOOTHED(-1, 7)) << 3) + ((SMOOTHED(-3, -9) < SMOOTHED(-5, 10)) << 2) + ((SMOOTHED(-13, 14) < SMOOTHED(17, -3)) << 1) + ((SMOOTHED(11, -19) < SMOOTHED(-1, -18)) << 0));
//		desc[25] = (uchar)(((SMOOTHED(8, -2) < SMOOTHED(-18, -23)) << 7) + ((SMOOTHED(0, -5) < SMOOTHED(-2, -9)) << 6) + ((SMOOTHED(-4, -11) < SMOOTHED(2, -8)) << 5) + ((SMOOTHED(14, 6) < SMOOTHED(-3, -6)) << 4) + ((SMOOTHED(-3, 0) < SMOOTHED(-15, 0)) << 3) + ((SMOOTHED(-9, 4) < SMOOTHED(-15, -9)) << 2) + ((SMOOTHED(-1, 11) < SMOOTHED(3, 11)) << 1) + ((SMOOTHED(-10, -16) < SMOOTHED(-7, 7)) << 0));
//		desc[26] = (uchar)(((SMOOTHED(-2, -10) < SMOOTHED(-10, -2)) << 7) + ((SMOOTHED(-5, -3) < SMOOTHED(5, -23)) << 6) + ((SMOOTHED(13, -8) < SMOOTHED(-15, -11)) << 5) + ((SMOOTHED(-15, 11) < SMOOTHED(6, -6)) << 4) + ((SMOOTHED(-16, -3) < SMOOTHED(-2, 2)) << 3) + ((SMOOTHED(6, 12) < SMOOTHED(-16, 24)) << 2) + ((SMOOTHED(-10, 0) < SMOOTHED(8, 11)) << 1) + ((SMOOTHED(-7, 7) < SMOOTHED(-19, -7)) << 0));
//		desc[27] = (uchar)(((SMOOTHED(5, 16) < SMOOTHED(9, -3)) << 7) + ((SMOOTHED(9, 7) < SMOOTHED(-7, -16)) << 6) + ((SMOOTHED(3, 2) < SMOOTHED(-10, 9)) << 5) + ((SMOOTHED(21, 1) < SMOOTHED(8, 7)) << 4) + ((SMOOTHED(7, 0) < SMOOTHED(1, 17)) << 3) + ((SMOOTHED(-8, 12) < SMOOTHED(9, 6)) << 2) + ((SMOOTHED(11, -7) < SMOOTHED(-8, -6)) << 1) + ((SMOOTHED(19, 0) < SMOOTHED(9, 3)) << 0));
//		desc[28] = (uchar)(((SMOOTHED(1, -7) < SMOOTHED(-5, -11)) << 7) + ((SMOOTHED(0, 8) < SMOOTHED(-2, 14)) << 6) + ((SMOOTHED(12, -2) < SMOOTHED(-15, -6)) << 5) + ((SMOOTHED(4, 12) < SMOOTHED(0, -21)) << 4) + ((SMOOTHED(17, -4) < SMOOTHED(-6, -7)) << 3) + ((SMOOTHED(-10, -9) < SMOOTHED(-14, -7)) << 2) + ((SMOOTHED(-15, -10) < SMOOTHED(-15, -14)) << 1) + ((SMOOTHED(-7, -5) < SMOOTHED(5, -12)) << 0));
//		desc[29] = (uchar)(((SMOOTHED(-4, 0) < SMOOTHED(15, -4)) << 7) + ((SMOOTHED(5, 2) < SMOOTHED(-6, -23)) << 6) + ((SMOOTHED(-4, -21) < SMOOTHED(-6, 4)) << 5) + ((SMOOTHED(-10, 5) < SMOOTHED(-15, 6)) << 4) + ((SMOOTHED(4, -3) < SMOOTHED(-1, 5)) << 3) + ((SMOOTHED(-4, 19) < SMOOTHED(-23, -4)) << 2) + ((SMOOTHED(-4, 17) < SMOOTHED(13, -11)) << 1) + ((SMOOTHED(1, 12) < SMOOTHED(4, -14)) << 0));
//		desc[30] = (uchar)(((SMOOTHED(-11, -6) < SMOOTHED(-20, 10)) << 7) + ((SMOOTHED(4, 5) < SMOOTHED(3, 20)) << 6) + ((SMOOTHED(-8, -20) < SMOOTHED(3, 1)) << 5) + ((SMOOTHED(-19, 9) < SMOOTHED(9, -3)) << 4) + ((SMOOTHED(18, 15) < SMOOTHED(11, -4)) << 3) + ((SMOOTHED(12, 16) < SMOOTHED(8, 7)) << 2) + ((SMOOTHED(-14, -8) < SMOOTHED(-3, 9)) << 1) + ((SMOOTHED(-6, 0) < SMOOTHED(2, -4)) << 0));
//		desc[31] = (uchar)(((SMOOTHED(1, -10) < SMOOTHED(-1, 2)) << 7) + ((SMOOTHED(8, -7) < SMOOTHED(-6, 18)) << 6) + ((SMOOTHED(9, 12) < SMOOTHED(-7, -23)) << 5) + ((SMOOTHED(8, -6) < SMOOTHED(5, 2)) << 4) + ((SMOOTHED(-9, 6) < SMOOTHED(-12, -7)) << 3) + ((SMOOTHED(-1, -2) < SMOOTHED(-7, 2)) << 2) + ((SMOOTHED(9, 9) < SMOOTHED(7, 15)) << 1) + ((SMOOTHED(6, 2) < SMOOTHED(-6, 6)) << 0));
//#undef SMOOTHED
//	}
//}
