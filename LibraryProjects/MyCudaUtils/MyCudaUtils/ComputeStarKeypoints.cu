#include "MyCudaUtils.h"
#include "Kernels.h"

static int
StarDetectorComputeResponses(const cv::Mat& img, cv::Mat& responses, cv::Mat& sizes, int maxSize);

void
computeIntegralImages(const cv::Mat& matI, cv::Mat& matS, cv::Mat& matT, cv::Mat& _FT)
{
	CV_Assert(matI.type() == CV_8U);

	int x, y, rows = matI.rows, cols = matI.cols;

	matS.create(rows + 1, cols + 1, CV_32S);
	matT.create(rows + 1, cols + 1, CV_32S);
	_FT.create(rows + 1, cols + 1, CV_32S);

	const uchar* I = matI.ptr<uchar>();
	int *S = matS.ptr<int>(), *T = matT.ptr<int>(), *FT = _FT.ptr<int>();
	int istep = (int)matI.step, step = (int)(matS.step / sizeof(S[0]));

	for (x = 0; x <= cols; x++)
		S[x] = T[x] = FT[x] = 0;

	S += step; T += step; FT += step;
	S[0] = T[0] = 0;
	FT[0] = I[0];
	for (x = 1; x < cols; x++)
	{
		S[x] = S[x - 1] + I[x - 1];
		T[x] = I[x - 1];
		FT[x] = I[x] + I[x - 1];
	}
	S[cols] = S[cols - 1] + I[cols - 1];
	T[cols] = FT[cols] = I[cols - 1];

	for (y = 2; y <= rows; y++)
	{
		I += istep, S += step, T += step, FT += step;

		S[0] = S[-step]; S[1] = S[-step + 1] + I[0];
		T[0] = T[-step + 1];
		T[1] = FT[0] = T[-step + 2] + I[-istep] + I[0];
		FT[1] = FT[-step + 2] + I[-istep] + I[1] + I[0];

		for (x = 2; x < cols; x++)
		{
			S[x] = S[x - 1] + S[-step + x] - S[-step + x - 1] + I[x - 1];
			T[x] = T[-step + x - 1] + T[-step + x + 1] - T[-step * 2 + x] + I[-istep + x - 1] + I[x - 1];
			FT[x] = FT[-step + x - 1] + FT[-step + x + 1] - FT[-step * 2 + x] + I[x] + I[x - 1];
		}

		S[cols] = S[cols - 1] + S[-step + cols] - S[-step + cols - 1] + I[cols - 1];
		T[cols] = FT[cols] = T[-step + cols - 1] + I[-istep + cols - 1] + I[cols - 1];
	}
}

struct StarFeature
{
	int area;
	int* p[8];
};

static int
StarDetectorComputeResponses(const cv::Mat& img, cv::Mat& responses, cv::Mat& sizes, int maxSize)
{
	const int MAX_PATTERN = 17;
	// The list of response sizes to compute
	static const int sizes0[] = { 1, 2, 3, 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128, -1 };
	// These pairs are the indexes in the sizes0 list above for the inner and outer diameters of the octogon to use 
	static const int pairs[][2] = { { 1, 0 }, { 3, 1 }, { 4, 2 }, { 5, 3 }, { 7, 4 }, { 8, 5 }, { 9, 6 },	// Outer and inner diameter of octogon
	{ 11, 8 }, { 13, 10 }, { 14, 11 }, { 15, 12 }, { 16, 14 }, { -1, -1 } };
	float invSizes[MAX_PATTERN][2];
	int sizes1[MAX_PATTERN];

	StarFeature f[MAX_PATTERN];

	cv::Mat sum, tilted, flatTilted;
	int rows = img.rows, cols = img.cols;
	int border, npatterns = 0, maxIdx = 0;

	CV_Assert(img.type() == CV_8UC1);

	responses.create(img.size(), CV_32F);
	sizes.create(img.size(), CV_16S);

	while (pairs[npatterns][0] >= 0 && !(sizes0[pairs[npatterns][0]] >= maxSize
		|| sizes0[pairs[npatterns + 1][0]] + sizes0[pairs[npatterns + 1][0]] / 2 >= std::min(rows, cols)))
	{
		++npatterns;
	}

	npatterns += (pairs[npatterns - 1][0] >= 0);
	maxIdx = pairs[npatterns - 1][0];


	//clock_t t1 = clock();
	computeIntegralImages(img, sum, tilted, flatTilted);
	//clock_t t2 = clock();
	//float diff = ((float)t2 - (float)t1);
	//std::cout << "Computing the integral images took " << diff/CLOCKS_PER_SEC << " seconds.";
	//std::cout << std::endl;

	int step = (int)(sum.step / sum.elemSize());

	// Eight parts of the integral images to check for each size
	int* starCnrs = new int[8 * (maxIdx + 1)];
	int* pstarCnrs = starCnrs;
	for (int i = 0; i <= maxIdx; i++)
	{
		int ur_size = sizes0[i], t_size = sizes0[i] + sizes0[i] / 2;
		int ur_area = (2 * ur_size + 1)*(2 * ur_size + 1);
		int t_area = t_size*t_size + (t_size + 1)*(t_size + 1);

		*pstarCnrs++ = (ur_size + 1)*step + ur_size + 1;
		*pstarCnrs++ = -ur_size*step + ur_size + 1;
		*pstarCnrs++ = (ur_size + 1)*step - ur_size;
		*pstarCnrs++ = -ur_size*step - ur_size;
		*pstarCnrs++ = (t_size + 1)*step + 1;
		*pstarCnrs++ = -t_size;
		*pstarCnrs++ = t_size + 1;
		*pstarCnrs++ = -t_size*step + 1;

		f[i].area = ur_area + t_area;
		sizes1[i] = sizes0[i];
	}
	// negate end points of the size range
	// for a faster rejection of very small or very large features in non-maxima suppression.
	sizes1[0] = -sizes1[0];
	sizes1[1] = -sizes1[1];
	sizes1[maxIdx] = -sizes1[maxIdx];

	// Create a border to not find features near the edges so we don't overlap
	border = sizes0[maxIdx] + sizes0[maxIdx] / 2;

	// Compute the area of each pattern for normalising the response
	float* invStarAreas = new float[2 * npatterns];
	int* host_pairs = new int[2 * npatterns];
	for (int i = 0; i < npatterns; i++)
	{
		int innerArea = f[pairs[i][1]].area;
		int outerArea = f[pairs[i][0]].area - innerArea;
		invSizes[i][0] = 1.f / outerArea;
		invSizes[i][1] = 1.f / innerArea;
		invStarAreas[i] = 1.f / outerArea;
		invStarAreas[i + npatterns] = 1.f / innerArea;
		host_pairs[i] = pairs[i][0];
		host_pairs[i + npatterns] = pairs[i][1];

	}
	
	responses = cv::Mat::zeros(responses.size(), CV_32F);
	sizes = cv::Mat::zeros(sizes.size(), CV_16S);

	int* D_intImage, *D_tiltImage, *D_flatTiltImage, *D_pairs, *D_starCnrs, *D_starSizes;
	float* D_responses, *D_invAreas;
	short* D_sizes;

	try {

		CUDA_CHECK(cudaSetDevice(0));

		CUDA_INIT_MEM(D_responses, responses.data, responses.rows * responses.cols * sizeof(float));

		CUDA_INIT_MEM(D_sizes, sizes.data, sizes.rows * sizes.cols * sizeof(short));

		CUDA_INIT_MEM(D_intImage, sum.data, sum.rows*sum.cols*sizeof(int));

		CUDA_INIT_MEM(D_tiltImage, tilted.data, tilted.rows * tilted.cols * sizeof(int));

		CUDA_INIT_MEM(D_flatTiltImage, flatTilted.data, flatTilted.rows * flatTilted.cols * sizeof(int));

		CUDA_INIT_MEM(D_pairs, host_pairs, npatterns * 2 * sizeof(int));

		CUDA_INIT_MEM(D_starSizes, sizes1, (maxIdx +1 )* sizeof(int));

		CUDA_INIT_MEM(D_invAreas, invStarAreas, npatterns * 2 * sizeof(float));

		CUDA_INIT_MEM(D_starCnrs, starCnrs, 8 * (maxIdx + 1)* sizeof(int));

		int blocks = cvCeil(responses.rows * responses.cols / 128.0f);
		StarDetectorKernel << <blocks, 128 >> >(D_intImage, D_tiltImage, D_flatTiltImage, maxIdx, D_starSizes, D_invAreas, D_starCnrs, npatterns, D_pairs, responses.rows, responses.cols, border, D_responses, D_sizes);

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_DOWNLOAD(responses.data, D_responses, responses.rows * responses.cols * sizeof(float));

		CUDA_DOWNLOAD(sizes.data, D_sizes, sizes.rows * sizes.cols * sizeof(short));

		throw (cudaSuccess);
	}
	catch (cudaError_t cudaStatus) {
		delete[] invStarAreas;
		delete[] starCnrs;
		delete[] host_pairs;
		cudaFree(D_intImage);
		cudaFree(D_tiltImage);
		cudaFree(D_flatTiltImage);
		cudaFree(D_pairs);
		cudaFree(D_starSizes);
		cudaFree(D_starCnrs);
		cudaFree(D_responses);
		cudaFree(D_invAreas);
		cudaFree(D_sizes);
	}


	return border;
}


// [in]  Pointers to the three computed integral images
// [in]  The number of different sized STAR shapes to be used
// [in]  sizes - the widths of each STAR shape to be used
// [in]  The inverse area of each STAR shape
// [in]  Coordinates defining the corner locations for each size STAR shape
// [in]  How many pairs of different sizes are being computed
// [in]  The indexes of the STARs that make up each pair (inner and outer star size)
// [in]  Number of rows and columns in all the images
__global__ void StarDetectorKernel(int* intImage, int* tiltImage, int* flatTiltImage, int numSizes, int* sizes, float* invAreas, int* starCnrs, int numPairs, int* pairs, int rows, int cols, int border, float* responses_out, short* sizes_out) {
	
	int pt_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int col = pt_idx % cols;
	int row = roundf((pt_idx - col) / cols);
	if (pt_idx >= rows*cols || row < border || row >= rows - border || col < border || col >= cols - border)
		return;
	
	int ofs = (cols + 1)*row + col;

	const int MAX_PATTERN = 17;
	int vals[MAX_PATTERN];
	float bestResponse = 0;
	int bestSize = 0;

	//patterns will have to be an array for each size it includes 8 offsets around 
	for (int sizeIdx = 0; sizeIdx <= numSizes; sizeIdx++)
	{
		vals[sizeIdx] = intImage[ofs + starCnrs[8 * sizeIdx + 0]] - intImage[ofs + starCnrs[8 * sizeIdx + 1]] - intImage[ofs + starCnrs[8 * sizeIdx + 2]] + intImage[ofs + starCnrs[8 * sizeIdx + 3]] +
			tiltImage[ofs + starCnrs[8 * sizeIdx + 4]] - flatTiltImage[ofs + starCnrs[8 * sizeIdx + 5]] - flatTiltImage[ofs + starCnrs[8 * sizeIdx + 6]] + tiltImage[ofs + starCnrs[8 * sizeIdx + 7]];
	}

	// Subtract inner response from outer and normalise. Take absolute value to get response magnitue
	// Find the largest response of all the sizes computed. 
	for (int i = 0; i < numPairs; i++)
	{
		int inner_sum = vals[pairs[i + numPairs]];
		int outer_sum = vals[pairs[i]] - inner_sum;
		float response = inner_sum*invAreas[i + numPairs] - outer_sum*invAreas[i];
		if (fabs(response) > fabs(bestResponse))
		{
			bestResponse = response;
			bestSize = sizes[pairs[i]];
		}
	}

	responses_out[pt_idx] = bestResponse;
	sizes_out[pt_idx] = (short)bestSize;

}

static bool StarDetectorSuppressLines(const cv::Mat& responses, const cv::Mat& sizes, cv::Point pt,
	int lineThresholdProjected, int lineThresholdBinarized)
{
	const float* r_ptr = responses.ptr<float>();
	int rstep = (int)(responses.step / sizeof(r_ptr[0]));
	const short* s_ptr = sizes.ptr<short>();
	int sstep = (int)(sizes.step / sizeof(s_ptr[0]));
	int sz = s_ptr[pt.y*sstep + pt.x];
	int x, y, delta = sz / 4, radius = delta * 4;
	float Lxx = 0, Lyy = 0, Lxy = 0;
	int Lxxb = 0, Lyyb = 0, Lxyb = 0;

	for (y = pt.y - radius; y <= pt.y + radius; y += delta)
	for (x = pt.x - radius; x <= pt.x + radius; x += delta)
	{
		float Lx = r_ptr[y*rstep + x + 1] - r_ptr[y*rstep + x - 1];
		float Ly = r_ptr[(y + 1)*rstep + x] - r_ptr[(y - 1)*rstep + x];
		Lxx += Lx*Lx; Lyy += Ly*Ly; Lxy += Lx*Ly;
	}

	if ((Lxx + Lyy)*(Lxx + Lyy) >= lineThresholdProjected*(Lxx*Lyy - Lxy*Lxy))
		return true;

	for (y = pt.y - radius; y <= pt.y + radius; y += delta)
	for (x = pt.x - radius; x <= pt.x + radius; x += delta)
	{
		int Lxb = (s_ptr[y*sstep + x + 1] == sz) - (s_ptr[y*sstep + x - 1] == sz);
		int Lyb = (s_ptr[(y + 1)*sstep + x] == sz) - (s_ptr[(y - 1)*sstep + x] == sz);
		Lxxb += Lxb * Lxb; Lyyb += Lyb * Lyb; Lxyb += Lxb * Lyb;
	}

	if ((Lxxb + Lyyb)*(Lxxb + Lyyb) >= lineThresholdBinarized*(Lxxb*Lyyb - Lxyb*Lxyb))
		return true;

	return false;
}



static void
StarDetectorSuppressNonmax(const cv::Mat& responses, const cv::Mat& sizes,
std::vector<cv::KeyPoint>& keypoints, int border,
int responseThreshold,
int lineThresholdProjected,
int lineThresholdBinarized,
int suppressNonmaxSize)
{
	int x, y, x1, y1, delta = suppressNonmaxSize / 2;
	int rows = responses.rows, cols = responses.cols;
	const float* r_ptr = responses.ptr<float>();
	int rstep = (int)(responses.step / sizeof(r_ptr[0]));
	const short* s_ptr = sizes.ptr<short>();
	int sstep = (int)(sizes.step / sizeof(s_ptr[0]));
	short featureSize = 0;

	for (y = border; y < rows - border; y += delta + 1)
	for (x = border; x < cols - border; x += delta + 1)
	{
		float maxResponse = (float)responseThreshold;
		float minResponse = (float)-responseThreshold;
		cv::Point maxPt(-1, -1), minPt(-1, -1);
		int tileEndY = MIN(y + delta, rows - border - 1);
		int tileEndX = MIN(x + delta, cols - border - 1);

		for (y1 = y; y1 <= tileEndY; y1++)
		for (x1 = x; x1 <= tileEndX; x1++)
		{
			float val = r_ptr[y1*rstep + x1];
			if (maxResponse < val)
			{
				maxResponse = val;
				maxPt = cv::Point(x1, y1);
			}
			else if (minResponse > val)
			{
				minResponse = val;
				minPt = cv::Point(x1, y1);
			}
		}

		if (maxPt.x >= 0)
		{
			for (y1 = maxPt.y - delta; y1 <= maxPt.y + delta; y1++)
			for (x1 = maxPt.x - delta; x1 <= maxPt.x + delta; x1++)
			{
				float val = r_ptr[y1*rstep + x1];
				if (val >= maxResponse && (y1 != maxPt.y || x1 != maxPt.x))
					goto skip_max;
			}

			if ((featureSize = s_ptr[maxPt.y*sstep + maxPt.x]) >= 4 &&
				!StarDetectorSuppressLines(responses, sizes, maxPt, lineThresholdProjected,
				lineThresholdBinarized))
			{
				cv::KeyPoint kpt((float)maxPt.x, (float)maxPt.y, featureSize, -1, maxResponse);
				keypoints.push_back(kpt);
			}
		}
	skip_max:
		if (minPt.x >= 0)
		{
			for (y1 = minPt.y - delta; y1 <= minPt.y + delta; y1++)
			for (x1 = minPt.x - delta; x1 <= minPt.x + delta; x1++)
			{
				float val = r_ptr[y1*rstep + x1];
				if (val <= minResponse && (y1 != minPt.y || x1 != minPt.x))
					goto skip_min;
			}

			if ((featureSize = s_ptr[minPt.y*sstep + minPt.x]) >= 4 &&
				!StarDetectorSuppressLines(responses, sizes, minPt,
				lineThresholdProjected, lineThresholdBinarized))
			{
				cv::KeyPoint kpt((float)minPt.x, (float)minPt.y, featureSize, -1, maxResponse);
				keypoints.push_back(kpt);
			}
		}
	skip_min:
		;
	}
}


cudaError_t MyCudaUtils::ComputeStarKeypoints(cv::Mat& image, std::vector<cv::KeyPoint>& kps, int maxSize, int responseThreshold,
	int lineThresholdProjected, int lineThresholdBinarized, int suppressNonmaxSize){
	cv::Mat responses, sizes;

	//clock_t t1 = clock();
	int border = StarDetectorComputeResponses(image, responses, sizes, maxSize);
	//clock_t t2 = clock();
	//float diff = ((float)t2 - (float)t1);
	//std::cout << "Computing the responses overall took " << diff / CLOCKS_PER_SEC << " seconds.";
	//std::cout << std::endl;
	kps.clear();
	if (border >= 0) {
		//t1 = t1 = clock();
		StarDetectorSuppressNonmax(responses, sizes, kps, border,
			responseThreshold, lineThresholdProjected,
			lineThresholdBinarized, suppressNonmaxSize);
		//t2 = clock();
		//float diff = ((float)t2 - (float)t1);
		//std::cout << "Performing non-max supression took " << diff / CLOCKS_PER_SEC << " seconds.";
		//std::cout << std::endl;
	}

	return cudaSuccess;
}




// ORIGINAL STAR DETECTOR RESPONSE FUNCTION
//static int
//StarDetectorComputeResponses(const cv::Mat& img, cv::Mat& responses, cv::Mat& sizes, int maxSize)
//{
//	const int MAX_PATTERN = 17;
//	// The list of response sizes to compute
//	static const int sizes0[] = { 1, 2, 3, 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128, -1 };
//	// These pairs are the indexes in the sizes0 list above for the inner and outer diameters of the octogon to use 
//	static const int pairs[][2] = { { 1, 0 }, { 3, 1 }, { 4, 2 }, { 5, 3 }, { 7, 4 }, { 8, 5 }, { 9, 6 },	// Outer and inner diameter of octogon
//	{ 11, 8 }, { 13, 10 }, { 14, 11 }, { 15, 12 }, { 16, 14 }, { -1, -1 } };
//	float invSizes[MAX_PATTERN][2];
//	int sizes1[MAX_PATTERN];
//
//#if CV_SSE2
//	__m128 invSizes4[MAX_PATTERN][2];
//	__m128 sizes1_4[MAX_PATTERN];
//	Cv32suf absmask;
//	absmask.i = 0x7fffffff;
//	volatile bool useSIMD = cv::checkHardwareSupport(CV_CPU_SSE2);
//#endif
//	StarFeature f[MAX_PATTERN];
//
//	cv::Mat sum, tilted, flatTilted;
//	int y, rows = img.rows, cols = img.cols;
//	int border, npatterns = 0, maxIdx = 0;
//
//	CV_Assert(img.type() == CV_8UC1);
//
//	responses.create(img.size(), CV_32F);
//	sizes.create(img.size(), CV_16S);
//
//	while (pairs[npatterns][0] >= 0 && !(sizes0[pairs[npatterns][0]] >= maxSize
//		|| sizes0[pairs[npatterns + 1][0]] + sizes0[pairs[npatterns + 1][0]] / 2 >= std::min(rows, cols)))
//	{
//		++npatterns;
//	}
//	// Figure out how many to use based upon the maximum response size specified and how big the actual image is. 
//	// Note if an arbitrary size is supplied it will not necesarily be used; there are 17 fixed size filters to use
//	// And the above code selects all the ones smaller than or equal in size to the supplied maxSize within the 17 available
//	// Note that pairs gives indices of sizes in the sizes0 to use - so maxIdx is how many sizes we actually go up to
//
//	npatterns += (pairs[npatterns - 1][0] >= 0);
//	maxIdx = pairs[npatterns - 1][0];
//
//	computeIntegralImages(img, sum, tilted, flatTilted);
//
//	int step = (int)(sum.step / sum.elemSize());
//
//	// Eight parts of the integral images to check for each size
//	for (int i = 0; i <= maxIdx; i++)
//	{
//		int ur_size = sizes0[i], t_size = sizes0[i] + sizes0[i] / 2;
//		// Area of the square components
//		int ur_area = (2 * ur_size + 1)*(2 * ur_size + 1);
//		// Area of the triangular components?
//		int t_area = t_size*t_size + (t_size + 1)*(t_size + 1);
//
//		// These are retangular points around centre
//		// Now these are pointers to what will have to be GPU memory (although we could just redfine them as an offset per thread
//		f[i].p[0] = sum.ptr<int>() + (ur_size + 1)*step + ur_size + 1;  // Okay so these pointers all point in what is effectively square and diamond shapes,
//		f[i].p[1] = sum.ptr<int>() - ur_size*step + ur_size + 1;		// except currently they are all located at the start of the image. The trick is that we
//		f[i].p[2] = sum.ptr<int>() + (ur_size + 1)*step - ur_size;		// Then add our actual pixel location to all of these pointers later in the process, and
//		f[i].p[3] = sum.ptr<int>() - ur_size*step - ur_size;			// they now point at square / triangle locations centred at the pixel of interest 
//
//		// Forms a diamond with the tilted patterns
//		f[i].p[4] = tilted.ptr<int>() + (t_size + 1)*step + 1;
//		f[i].p[5] = flatTilted.ptr<int>() - t_size;
//		f[i].p[6] = flatTilted.ptr<int>() + t_size + 1;
//		f[i].p[7] = tilted.ptr<int>() - t_size*step + 1;
//
//		// So I think the star shape is just the addition of the two squares rather than the exact star shape
//		// so the overlap is present and also not cared about - it probably makes a nice ring effect with 
//		// pixels inside the overlap octogon scaled by two. I don't see why the flatTilted one is needed though
//		// at this stage, it seems to be doing exactly the same job as the tilted
//
//		f[i].area = ur_area + t_area;
//		sizes1[i] = sizes0[i];
//	}
//	// negate end points of the size range
//	// for a faster rejection of very small or very large features in non-maxima suppression.
//	sizes1[0] = -sizes1[0];
//	sizes1[1] = -sizes1[1];
//	sizes1[maxIdx] = -sizes1[maxIdx];
//	// Create a border to not find features near the edges so we don't overlap
//	border = sizes0[maxIdx] + sizes0[maxIdx] / 2;
//
//	// Compute the area of each pattern for normalising the response
//	// 
//	for (int i = 0; i < npatterns; i++)
//	{
//		int innerArea = f[pairs[i][1]].area;
//		int outerArea = f[pairs[i][0]].area - innerArea;
//		invSizes[i][0] = 1.f / outerArea;
//		invSizes[i][1] = 1.f / innerArea;
//	}
//
//#if CV_SSE2
//	if (useSIMD)
//	{
//		for (int i = 0; i < npatterns; i++)
//		{
//			_mm_store_ps((float*)&invSizes4[i][0], _mm_set1_ps(invSizes[i][0]));
//			_mm_store_ps((float*)&invSizes4[i][1], _mm_set1_ps(invSizes[i][1]));
//		}
//
//		for (int i = 0; i <= maxIdx; i++)
//			_mm_store_ps((float*)&sizes1_4[i], _mm_set1_ps((float)sizes1[i]));
//	}
//#endif
//
//	// Initialise to zeroes? Don't we have Mat::zeros();
//		responses = cv::Mat::zeros(responses.size(), CV_32F);
//		sizes = cv::Mat::zeros(sizes.size(), CV_16S);
//	//for (y = 0; y < border; y++)
//	//{
//	//	float* r_ptr = responses.ptr<float>(y);
//	//	float* r_ptr2 = responses.ptr<float>(rows - 1 - y);
//	//	short* s_ptr = sizes.ptr<short>(y);
//	//	short* s_ptr2 = sizes.ptr<short>(rows - 1 - y);
//
//	//	memset(r_ptr, 0, cols*sizeof(r_ptr[0])); // Initialise block of memory to same value of 0, here its the whole row
//	//	memset(r_ptr2, 0, cols*sizeof(r_ptr2[0])); // So we basically initialise the responses and sizes arrays to zero?
//	//	memset(s_ptr, 0, cols*sizeof(s_ptr[0]));
//	//	memset(s_ptr2, 0, cols*sizeof(s_ptr2[0]));
//	//}
//
//	// Now process inside the borders, from row to row (y)
//	// and col to col (x)
//	// This is the part that is completely parallelisable - its a reasonably complex operation on every pixel. 
//	for (y = border; y < rows - border; y++)
//	{
//		int x = border;
//		float* r_ptr = responses.ptr<float>(y);
//		short* s_ptr = sizes.ptr<short>(y);
//
//		// Now initialise the borders to zero again? Why all this zero setting?
//		// Maybe its faster than Mat::zeroes();
//		//memset(r_ptr, 0, border*sizeof(r_ptr[0]));
//		//memset(s_ptr, 0, border*sizeof(s_ptr[0]));
//		//memset(r_ptr + cols - border, 0, border*sizeof(r_ptr[0]));
//		//memset(s_ptr + cols - border, 0, border*sizeof(s_ptr[0]));
//
//#if CV_SSE2
//		if (useSIMD)
//		{
//			__m128 absmask4 = _mm_set1_ps(absmask.f);
//			for (; x <= cols - border - 4; x += 4)
//			{
//				int ofs = y*step + x;
//				__m128 vals[MAX_PATTERN];
//				__m128 bestResponse = _mm_setzero_ps();
//				__m128 bestSize = _mm_setzero_ps();
//
//				for (int i = 0; i <= maxIdx; i++)
//				{
//					const int** p = (const int**)&f[i].p[0];
//					__m128i r0 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[0] + ofs)),
//						_mm_loadu_si128((const __m128i*)(p[1] + ofs)));
//					__m128i r1 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[3] + ofs)),
//						_mm_loadu_si128((const __m128i*)(p[2] + ofs)));
//					__m128i r2 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[4] + ofs)),
//						_mm_loadu_si128((const __m128i*)(p[5] + ofs)));
//					__m128i r3 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[7] + ofs)),
//						_mm_loadu_si128((const __m128i*)(p[6] + ofs)));
//					r0 = _mm_add_epi32(_mm_add_epi32(r0, r1), _mm_add_epi32(r2, r3));
//					_mm_store_ps((float*)&vals[i], _mm_cvtepi32_ps(r0));
//				}
//
//				for (int i = 0; i < npatterns; i++)
//				{
//					__m128 inner_sum = vals[pairs[i][1]];
//					__m128 outer_sum = _mm_sub_ps(vals[pairs[i][0]], inner_sum);
//					__m128 response = _mm_sub_ps(_mm_mul_ps(inner_sum, invSizes4[i][1]),
//						_mm_mul_ps(outer_sum, invSizes4[i][0]));
//					__m128 swapmask = _mm_cmpgt_ps(_mm_and_ps(response, absmask4),
//						_mm_and_ps(bestResponse, absmask4));
//					bestResponse = _mm_xor_ps(bestResponse,
//						_mm_and_ps(_mm_xor_ps(response, bestResponse), swapmask));
//					bestSize = _mm_xor_ps(bestSize,
//						_mm_and_ps(_mm_xor_ps(sizes1_4[pairs[i][0]], bestSize), swapmask));
//				}
//
//				_mm_storeu_ps(r_ptr + x, bestResponse);
//				_mm_storel_epi64((__m128i*)(s_ptr + x),
//					_mm_packs_epi32(_mm_cvtps_epi32(bestSize), _mm_setzero_si128()));
//			}
//		}
//#endif
//		// Note if SSE is enabled, then the previous will excute and x = cols - border so this loop will never enter
//		// Note that this serial implementation is actually much closer to the equivalent GPU implementation - except
//		// instead of looping we just launch one thread per pixel 
//		// For all the columns
//		for (; x < cols - border; x++)
//		{
//			// Everything from here is the individual pixel processing we need. 
//			int ofs = y*step + x;
//			int vals[MAX_PATTERN];
//			float bestResponse = 0;
//			int bestSize = 0;
//
//			// For all the sizes compute the response
//			for (int i = 0; i <= maxIdx; i++)
//			{
//				const int** p = (const int**)&f[i].p[0];
//				vals[i] = p[0][ofs] - p[1][ofs] - p[2][ofs] + p[3][ofs] +
//					p[4][ofs] - p[5][ofs] - p[6][ofs] + p[7][ofs];
//			}
//
//			// Subtract inner response from outer and normalise. Take absolute value to get response magnitue
//			// Find the largest response of all the sizes computed. 
//			for (int i = 0; i < npatterns; i++)
//			{
//				int inner_sum = vals[pairs[i][1]];
//				int outer_sum = vals[pairs[i][0]] - inner_sum;
//				float response = inner_sum*invSizes[i][1] - outer_sum*invSizes[i][0];
//				if (fabs(response) > fabs(bestResponse))
//				{
//					bestResponse = response;
//					bestSize = sizes1[pairs[i][0]];
//				}
//			}
//
//			r_ptr[x] = bestResponse;
//			s_ptr[x] = (short)bestSize;
//		}
//	}
//
//	return border;
//}
//
