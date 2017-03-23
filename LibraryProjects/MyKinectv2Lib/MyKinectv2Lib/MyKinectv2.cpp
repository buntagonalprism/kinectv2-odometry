#include "MyKinectv2.h"



/*********************************************************************************
*					  C O N -  +   D E - S T R U C T O R 
*********************************************************************************/

// Constructor for Kinectv2 object
// Allocates memory for image buffers
// Hard-codes the intrinsic and extrinsic camera parameters
MyKinectv2::MyKinectv2() :
multiReader(NULL),
colourReader(NULL),
depthReader(NULL),
irReader(NULL),
kinect(NULL)
{

	m_pDepthRGBX = new RGBQUAD[cDepthWidth * cDepthHeight];
	m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];
	m_pIrRGBX = new RGBQUAD[cDepthWidth * cDepthHeight];
	m_pColourGreyscale = new BYTE[cColorWidth * cColorHeight];

}


// Destructor for Kinectv2 object 
// Performs memroy deallocation for image buffers
MyKinectv2::~MyKinectv2() {

	// Free the memory
	if (m_pDepthRGBX)
	{
		delete[] m_pDepthRGBX;
		m_pDepthRGBX = NULL;
	}

	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}

	if (m_pIrRGBX)
	{
		delete[] m_pIrRGBX;
		m_pIrRGBX = NULL;
	}

	if(m_pColourGreyscale)
	{
		delete[] m_pColourGreyscale;
		m_pColourGreyscale = NULL;
	}

	// Release and close other things
	SafeRelease(multiReader);
	SafeRelease(colourReader);
	SafeRelease(depthReader);
	SafeRelease(irReader);
	if (kinect)
		kinect->Close();
	SafeRelease(kinect);

}


/*********************************************************************************
*		                 D A T A    C A P T U R E  
*********************************************************************************/

// Connects to the default Kinectv2 sensor for synchronised frame capture
// [in]  The folder in which data from the kinect should be saved
// [in]  Which streams to capture data from with the Update() command
void MyKinectv2::InitSyncCapture(char* folderDirectory, int sourceTypes, bool greyscale) {
	
	strcpy(folderName, folderDirectory);
	boost::filesystem::path dir(folderName);
	if (boost::filesystem::create_directory(dir)) 
		std::cout << "Folder: " << folderName << " created" << "\n";

	capType = SYNC_CAP;
	greyFlag = greyscale;
	frameTypes = sourceTypes;

	HRESULT hr = GetDefaultKinectSensor(&kinect);
	
	if (FAILED(hr))
		std::cout << "Could not find a default Kinectv2 sensor" << std::endl;

	if (kinect) {						// makes sure null pointer not returned
		hr = kinect->Open();
		if (SUCCEEDED(hr))
			kinect->OpenMultiSourceFrameReader(sourceTypes, &multiReader);
	}
}


// Connects to default Kinectv2 sensor for asynchronous frame capture
// [in]  The folder in which data from the kinect should be saved
// [in]  Which streams to capture data from with the Update() command
// [in]  Whether colour images should be greyscale. Reduces data size by 1/4 but may slow capture
void MyKinectv2::InitAsyncCapture(char* folderDirectory, int sourceTypes, bool greyscale ) {
	
	strcpy(folderName, folderDirectory);
	boost::filesystem::path dir(folderName);
	if (boost::filesystem::create_directory(dir)) 
		std::cout << "Folder: " << folderName << " created" << "\n";
	
	capType = ASYNC_CAP;
	greyFlag = greyscale;
	frameTypes = sourceTypes;

	IColorFrameSource* colourSource = NULL;
	IDepthFrameSource* depthSource = NULL;
	IInfraredFrameSource* irSource = NULL;
	
	// A windows-specific error result type - note its just typedef as a long
	HRESULT hr = GetDefaultKinectSensor(&kinect);

	// inline function FAILED defined as hr < 0
	if (FAILED(hr))						
		std::cout << "Could not find a default Kinectv2 sensor" << std::endl;

	if (kinect) {						// makes sure null pointer not returned
		hr = kinect->Open();

		if ((sourceTypes & FrameSourceTypes_Color) > 0) {
			kinect->get_ColorFrameSource(&colourSource);
			colourSource->OpenReader(&colourReader);
		}
		if ((sourceTypes & FrameSourceTypes_Depth) > 0) {
			kinect->get_DepthFrameSource(&depthSource);
			depthSource->OpenReader(&depthReader);
		}
		if ((sourceTypes & FrameSourceTypes_Infrared) > 0) {
			kinect->get_InfraredFrameSource(&irSource);
			irSource->OpenReader(&irReader);
		}
	}

	if (!kinect || FAILED(hr)) {
		std::cout << "Something went wrong during Kinectv2 initialisation" << std::endl;
	}

	SafeRelease(colourSource);
	SafeRelease(depthSource);
	SafeRelease(irSource);
}


// Poll this function to capture data of the initialised streams
// [ret] 0 if no data available, or the number of frame types saved if any data captured
//       No guarantee that all frame types initialised will be captured each update
int MyKinectv2::GrabFrames(void) {
	HRESULT hr = E_PENDING, hrc = E_PENDING, hrd = E_PENDING, hri = E_PENDING;
	int frames = 0;
	switch (capType) {
	// Asynchronous read means checking each of the active frame readers individually
	// Consider use of timestamping each one when there are multiple streams available
	case ASYNC_CAP:
		if (frameTypes & FrameSourceTypes_Color) {
			hr = colourReader->AcquireLatestFrame(&colourFrame);
			if (SUCCEEDED(hr)) {
				timer.saveTime();
				SaveColour(colourFrame);
				std::cout << "Colour saved" << std::endl;
				frames++;
			}
		}

		if (frameTypes & FrameSourceTypes_Depth) {
			hr = depthReader->AcquireLatestFrame(&depthFrame);
			if (SUCCEEDED(hr)) {
				timer.saveTime();
				SaveDepth(depthFrame);
				std::cout << "Depth saved" << std::endl;
				frames++;
			}
		}
		if (frameTypes & FrameSourceTypes_Infrared) {
			hr = irReader->AcquireLatestFrame(&irFrame);
			if (SUCCEEDED(hr)) {
				timer.saveTime();
				SaveIR(irFrame);
				std::cout << "Depth saved" << std::endl;
				frames++;
			}
		}
		
		break;

	// Synchronous read acquires a multiframe then gets individual frame references from it
	case SYNC_CAP:
		hr = multiReader->AcquireLatestFrame(&multiFrame);
		if (SUCCEEDED(hr)) {
			timer.saveTime();
			timer.toc("Time between frame captures: ");
			timer.tic();
			// We need to get all the frame references first else the internal
			// buffers will start to be overridden by the time we have finished
			// saving the first FrameSourceType. 
			if (frameTypes & FrameSourceTypes_Color) {
				multiFrame->get_ColorFrameReference(&colourRef);
				hrc = colourRef->AcquireFrame(&colourFrame);
			}
			if (frameTypes & FrameSourceTypes_Depth) {
				multiFrame->get_DepthFrameReference(&depthRef);
				hrd = depthRef->AcquireFrame(&depthFrame);
			}
			if (frameTypes & FrameSourceTypes_Infrared) {
				multiFrame->get_InfraredFrameReference(&irRef);
				hri = irRef->AcquireFrame(&irFrame);
			}
			timer.toc("Frame references acquired");

			// Now process the frames we have acquired
			
			if (SUCCEEDED(hrc)) {
				timer.tic();
				SaveColour(colourFrame);
				timer.toc("Colour saved");
				frames++;
				std::cout << "Colour saved, ";
			}
			

			if (SUCCEEDED(hrd)) {
				timer.tic();
				SaveDepth(depthFrame);
				timer.toc("Depth saved");
				frames++;
				std::cout << "Depth saved, ";
			}
			

			if (SUCCEEDED(hri)) {
				SaveIR(irFrame);
				frames++;
				std::cout << "IR saved, ";
			}
			
			std::cout << "synchronously." << std::endl;
		}
		break;
	}
	return frames;

}

// Begins data capture
void MyKinectv2::QuickSyncCapture(void) {

	std::cout << "Data capture beginning, press q to quit" << std::endl;
	timer.tic();
	HRESULT hr = E_PENDING, hrc = E_PENDING, hrd = E_PENDING;
	bool quit = false;
	while (!quit)
	{
		while (_kbhit()) {
			char hit = _getch();
			if (hit == 27)
				quit = true;
		}
		this->GrabFrames();
		//	timer.toc("Time between frame captures");
		//	timer.tic();
		
		hr = multiReader->AcquireLatestFrame(&multiFrame);
		if (SUCCEEDED(hr)) {
			timer.saveTime();
			timer.toc("Oops I captured them here");
			multiFrame->get_ColorFrameReference(&colourRef);
			multiFrame->get_DepthFrameReference(&depthRef);

			hrc = colourRef->AcquireFrame(&colourFrame);
			hrd = depthRef->AcquireFrame(&depthFrame);
			
			if (SUCCEEDED(hrc)) {
				SaveColour(colourFrame);
				std::cout << "Colour Saved: ";
			}
			
			if (SUCCEEDED(hrd)) {
				SaveDepth(depthFrame);
				std::wcout << "Depth Saved: ";
			}

			std::cout << "In this operation" << std::endl;

		}
	}
}


// Begins data capture
void MyKinectv2::QuickAsyncCapture(void) {

	std::cout << "Data capture beginning, press q to quit" << std::endl;
	HRESULT hr;
	bool quit = false;
	while (!quit)
	{
		while (_kbhit()) {
			char hit = _getch();
			if (hit == 27)
				quit = true;
		}

		hr = colourReader->AcquireLatestFrame(&colourFrame);
		if (SUCCEEDED(hr)) {
			timer.saveTime();
			SaveColour(colourFrame);
			std::cout << "Colour saved" << std::endl;
		}

		hr = depthReader->AcquireLatestFrame(&depthFrame);
		if (SUCCEEDED(hr)) {
			timer.saveTime();
			SaveDepth(depthFrame);
			std::cout << "Depth saved" << std::endl;
		}

	}
}




/*********************************************************************************
*					    S A V I N G    I M A G E S
*********************************************************************************/

// Saves a single image of the given types and file name
// [in]  The stream source to save. Requires previous InitXXXXCapture call
// [in]  The name of the image to save
//       Use NULL for a timestamp to be used as the name appended with image type ('i', 'c', or 'd')
//       Colour images are saved as raw 8-bit RGB name.bmp
//       Depth images are saved raw as 16-bit greyscale name.pgm
//       IR images are scaled between min and max values as 8-bit name.png
// [ret] True if sync capture type and all frames requested were successfully captured,
//       otherwise false. For async always returns true
bool MyKinectv2::SaveImage(int sourceTypes_in, const char* imageName_in ) {
	HRESULT hr;
	switch (capType) {
	case ASYNC_CAP:
		// Acquire all the frames first to minimise latency between them
		if ((sourceTypes_in & FrameSourceTypes_Color) > 0)
			while (FAILED(colourReader->AcquireLatestFrame(&colourFrame)));

		if ((sourceTypes_in & FrameSourceTypes_Depth) > 0)
			while (FAILED(depthReader->AcquireLatestFrame(&depthFrame)));

		if ((sourceTypes_in & FrameSourceTypes_Infrared) > 0)
			while (FAILED(irReader->AcquireLatestFrame(&irFrame)));

		// Then save all the frames
		if ((sourceTypes_in & FrameSourceTypes_Depth) > 0)
			SaveDepth(depthFrame, imageName_in);
		
		if ((sourceTypes_in & FrameSourceTypes_Color) > 0)
			SaveColour(colourFrame, imageName_in);

		if ((sourceTypes_in & FrameSourceTypes_Infrared) > 0)
			SaveIR(irFrame, imageName_in);

		return true;
		break;

	case SYNC_CAP:
		while (FAILED(multiReader->AcquireLatestFrame(&multiFrame)));

		if ((frameTypes & FrameSourceTypes_Color) > 0) {
			multiFrame->get_ColorFrameReference(&colourRef);
			hr = colourRef->AcquireFrame(&colourFrame);
			if (SUCCEEDED(hr))
				SaveColour(colourFrame, imageName_in);
		}

		if ((frameTypes & FrameSourceTypes_Depth) > 0) {
			multiFrame->get_DepthFrameReference(&depthRef);
			hr = depthRef->AcquireFrame(&depthFrame);
			if (SUCCEEDED(hr))
				SaveDepth(depthFrame, imageName_in);
		}

		if ((frameTypes & FrameSourceTypes_Infrared) > 0) {
			multiFrame->get_InfraredFrameReference(&irRef);
			hr = irRef->AcquireFrame(&irFrame);
			if (SUCCEEDED(hr))
				SaveIR(irFrame, imageName_in);
		}
		// Work out conditions to return true here
		break;
	}
	return false;
}


// Saves a single colour image of the given name
void MyKinectv2::SaveColourImage(const char* imageName) {
	while (FAILED(colourReader->AcquireLatestFrame(&colourFrame)));
	SaveColour(colourFrame, imageName);
}


// Saves a single depth image of the given file name
void MyKinectv2::SaveDepthImage(const char* imageName) {
	while (FAILED(depthReader->AcquireLatestFrame(&depthFrame)));
	SaveDepth(depthFrame, imageName);
}


// Saves a single IR image of the given file name
void MyKinectv2::SaveIRImage(const char* imageName) {
	while (FAILED(irReader->AcquireLatestFrame(&irFrame)));
	SaveIR(irFrame, imageName);
}


// Saves synchronised depth and colour images of a given file name
// [in]  The name of the images to save - one will be name.bmp, the other name.pgm
//       Use NULL for a timestamp to be used as the name
void MyKinectv2::SaveDepthAndColourImage(const char* imageName) {
	while (FAILED(multiReader->AcquireLatestFrame(&multiFrame)));
	multiFrame->get_ColorFrameReference(&colourRef);
	multiFrame->get_DepthFrameReference(&depthRef);

	HRESULT hr = colourRef->AcquireFrame(&colourFrame);
	if (SUCCEEDED(hr))
		SaveColour(colourFrame, imageName);

	hr = depthRef->AcquireFrame(&depthFrame);
	if (SUCCEEDED(hr))
		SaveDepth(depthFrame, imageName);
}




/*********************************************************************************
*					     S T R E A M     V I E W E R
*********************************************************************************/
// Starts a viewer for the frame source 
// [in]  The frame source type to start viewing
void MyKinectv2::StartViewer(int sourceTypes_in, int depth_min_distance, int depth_max_distance) {
	if (capType != ASYNC_CAP) {
		std::cout << "Synchronised viewer not currently implemented" << std::endl;
		return;
	}
	//cv::namedWindow("Depth Stream Viewer");
	cv::Mat image;
	bool quit = false;
	bool screenshot = false;
	int viewerType = sourceTypes_in;
	char hit = 0;
	HRESULT hr;
	while (!quit)
	{
		while (_kbhit()) 
			char hit = _getch();
			
		switch (viewerType) {
		case FrameSourceTypes_Color:
			hr = colourReader->AcquireLatestFrame(&colourFrame);
			if (SUCCEEDED(hr)) {
				IColorCameraSettings* camSettings;
				colourFrame->get_ColorCameraSettings(&camSettings);
				TIMESPAN exposureTime;
				camSettings->get_ExposureTime(&exposureTime);
				std::cout << "Exposure time: " << exposureTime << std::endl;
				std::cout << "Colour frame viewer currently unimplemented" << std::endl;
				SafeRelease(colourFrame);	// Don't forget to put this in the handling function
			}
			break;
		case FrameSourceTypes_Infrared:
			std::cout << "IR frame viewer currently unimplemented" << std::endl;
			break;

		case FrameSourceTypes_Depth:
			hr = depthReader->AcquireLatestFrame(&depthFrame);
			if (SUCCEEDED(hr)) {
				DrawDepth(depthFrame, image, depth_min_distance, depth_max_distance);
				cv::imshow("Depth Stream Viewer", image);
				hit = cv::waitKey(10);
			}
			break;
		}

		// Check the input from any source

		// 'q' or 'esc' to quit
		if (hit == 27 || hit == 'q')
			quit = true;

		// 's' for screenshot
		if (hit == 's') {
			std::cout << "Enter file name for screenshot: ";
			string fileName;
			std::cin >> fileName; 
			
			switch (viewerType) {
			case FrameSourceTypes_Depth:
				sprintf_s(depthFileName, "%s/%s.bmp", folderName, fileName.c_str());
				imwrite(depthFileName, image);
				break;

				// Add other cases with other viewers here
			}
		}

		// Add options for switching the vieiwer while it is running
		if (hit == 'c') {}
	}
	cv::destroyAllWindows();
}



/*********************************************************************************
*					    C A M E R A    P A R A M E T E R S 
*********************************************************************************/

// Allows a different set of depth intrinsic parameters to be associated with the object
// [in]  The intrinsics of focal length and camera centre in pixels
void MyKinectv2::SetDepthIntrinsics(float& fx, float& fy, float& cx, float& cy) {
	camParams.depth_fx = fx; 
	camParams.depth_fy = fy;
	camParams.depth_cx = cx;
	camParams.depth_cy = cy;
}


// Allows a different set of colour intrinsic parameters to be associated with the object
// [in]  The intrinsics of focal length and camera centre in pixels
void MyKinectv2::SetColourIntrinsics(float& fx, float& fy, float& cx, float& cy) {
	camParams.colour_fx = fx;
	camParams.colour_fy = fy;
	camParams.colour_cx = cx;
	camParams.colour_cy = cy;
}

// Allows a different set of extrinsic parameters between the two sensors to be associated with the object
// [in]  The homogenous transformation matrix between the two sensors in mm
void MyKinectv2::SetSensorRelativeTransform(Eigen::Matrix4f relativeTransform) {
	camParams.depthToColourTransform = relativeTransform;
}


// Returns the camera parameters of the current object
// [ret] The KinectCamParams struct of current parameters
KinectCamParams MyKinectv2::GetCameraParameters(void) {
	return camParams;
}



/*********************************************************************************
*		   P R I V A T E    F U N C S :   D R A W I N G    F R A M E S
*********************************************************************************/

// Draws a depth frame scaled between min and max for 'jet' colourmap
void MyKinectv2::DrawDepth(IDepthFrame* pDepthFrame, cv::Mat& depthImage_out, int min_depth, int max_depth) {

	Mat depthImage, depthScaled, depthColoured;

	FrameToImageD(pDepthFrame, depthImage);

	depthScaled.create(depthImage.rows, depthImage.cols, CV_8UC1);

	for (int row = 0; row < depthImage.rows; row++) {
		for (int col = 0; col < depthImage.cols; col++) {
			unsigned short depth = depthImage.at<unsigned short>(row, col);
			if (depth >= min_depth && depth <= max_depth) {
				unsigned char depthValue = 255 * (float)(depth - min_depth) / (max_depth - min_depth);
				depthScaled.at<unsigned char>(row, col) = depthValue;
			}
			else if (depth >= max_depth)
				depthScaled.at<unsigned char>(row, col) = 255;
			else
				depthScaled.at<unsigned char>(row, col) = 0;
		}
	}

	cv::applyColorMap(depthScaled, depthImage_out, cv::COLORMAP_JET);

	SafeRelease(pDepthFrame);
}


/*********************************************************************************
*	    P R I V A T E    F U N C S :   C O N V E R T I N G    F R A M E S   
*********************************************************************************/

// Converts a Kinectv2 Depth Frame to a unsigned short cv::Mat 
void MyKinectv2::FrameToImageD(IDepthFrame* pDepthFrame, Mat& depthImage) {
	IFrameDescription* pFrameDescription = NULL;
	int nWidth = 0;
	int nHeight = 0;
	UINT nBufferSize = 0;
	UINT16 *pBuffer = NULL;
	pDepthFrame->get_FrameDescription(&pFrameDescription);
	pFrameDescription->get_Height(&nHeight);
	pFrameDescription->get_Width(&nWidth);

	depthImage.create(nHeight, nWidth, CV_16UC1);

	pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);

	for (int y = 0; y < nHeight; ++y) {
		USHORT* pixelrow = depthImage.ptr<USHORT>(y);
		for (int x = 0; x < nWidth; ++x) {
			USHORT depth = *pBuffer;
			if (pBuffer != 0)
				pixelrow[x] = depth;
			else
				pixelrow[x] = 0;
			++pBuffer;
		}
	}
}

// Converts a Kinectv2 Infrard Frame to a unsigned short cv::Mat 
void MyKinectv2::FrameToImageI(IInfraredFrame* pIRFrame, Mat& irImage) {
	IFrameDescription* pFrameDescription = NULL;
	int nWidth = 0;
	int nHeight = 0;
	UINT nBufferSize = 0;
	UINT16 *pBuffer = NULL;
	pIRFrame->get_FrameDescription(&pFrameDescription);
	pFrameDescription->get_Height(&nHeight);
	pFrameDescription->get_Width(&nWidth);

	irImage.create(nHeight, nWidth, CV_16UC1);

	pIRFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);

	for (int y = 0; y < nHeight; ++y) {
		USHORT* pixelrow = irImage.ptr<USHORT>(y);
		for (int x = 0; x < nWidth; ++x) {
			USHORT depth = *pBuffer;
			if (pBuffer != 0)
				pixelrow[x] = depth;
			else
				pixelrow[x] = 0;
			++pBuffer;
		}
	}
}


/*********************************************************************************
*		    P R I V A T E    F U N C S :   S A V I N G   I M A G E S
*********************************************************************************/

// Handles a colour frame
void MyKinectv2::SaveColour(IColorFrame* pColorFrame, const char* fileName) {
	
	HRESULT hr;
	INT64 nTime = 0;
	IFrameDescription* pFrameDescription = NULL;
	int nWidth = 0;
	int nHeight = 0;
	ColorImageFormat imageFormat = ColorImageFormat_None;
	UINT nBufferSize = 0;
	RGBQUAD *pBuffer = NULL;
	BYTE* pGreyBuff = NULL;
	BYTE* pYuvBuffer = NULL;
	cv::Mat greyImage;

	pColorFrame->get_RelativeTime(&nTime);

	pColorFrame->get_FrameDescription(&pFrameDescription);
		
	pFrameDescription->get_Width(&nWidth);
		
	pFrameDescription->get_Height(&nHeight);
		
	pColorFrame->get_RawColorImageFormat(&imageFormat);
		
	if (imageFormat == ColorImageFormat_Bgra)
	{
		// Get RGBA Buffer directly
		hr = pColorFrame->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE**>(&pBuffer));
	}
	else if (m_pColorRGBX && greyFlag == false)
	{
		// Get and convert the YUYV buffer to RGB
		pBuffer = m_pColorRGBX;
		nBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
		hr = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(pBuffer), ColorImageFormat_Bgra);
	}
	else if (m_pColorRGBX && greyFlag == true) {

		// Get the YUYV buffer
		UINT nYubBufSize = 0;
		hr = pColorFrame->AccessRawUnderlyingBuffer(&nYubBufSize, &pYuvBuffer);
	
		// Copy luminance values (every second byte) to greyscale buffer
		BYTE* pYuvBytePtr = pYuvBuffer;
		pGreyBuff = m_pColourGreyscale;
		for (int i = 0; i < cColorHeight*cColorWidth; i++) {
			*(pGreyBuff++) = pYuvBytePtr[2 * i];
		}
	}
	else
		hr = E_FAIL;
	
	// Create the filename
	if (fileName == NULL)
		sprintf_s(colourFileName, "%s/%I64dc.bmp", folderName, timer.printTime());
	else
		sprintf_s(colourFileName, "%s/%s.bmp", folderName, fileName);

	// Save the buffer to a file
	if (greyFlag) {
		cv::Mat wrapper(cColorHeight, cColorWidth, CV_8UC1, m_pColourGreyscale, cColorWidth);
		cv::imwrite(colourFileName, wrapper);
	}
	else {
		// I'd like to replace this with an OpenCV function for consistency, but it'll do for now
		size_t size = strlen(colourFileName) + 1;

		wchar_t* wideName = new wchar_t[size];

		mbstowcs(wideName, colourFileName, size);

		SaveColourBitmapToFile(reinterpret_cast<BYTE*>(pBuffer), nWidth, nHeight, sizeof(RGBQUAD)* 8, wideName);
	}
		
	// These are important to capture the next image
	SafeRelease(pFrameDescription);
	
	SafeRelease(pColorFrame);
}


// Handles a depth image by writing it to the designated folder with timestamp
void MyKinectv2::SaveDepth(IDepthFrame* pDepthFrame, const char* fileName) {
	
	Mat depthImage;

	FrameToImageD(pDepthFrame, depthImage);

	if (fileName == NULL)
		sprintf_s(depthFileName, "%s/%I64dd.pgm", folderName, timer.printTime());
	else
		sprintf_s(depthFileName, "%s/%s.pgm", folderName, fileName);

	imwrite(depthFileName, depthImage);

	SafeRelease(pDepthFrame);
}

// Handles a depth image by writing it to the designated folder with timestamp
void MyKinectv2::SaveIR(IInfraredFrame* pIrFrame, const char* fileName) {

	Mat irImage;

	FrameToImageI(pIrFrame, irImage);

	if (fileName == NULL)
		sprintf_s(depthFileName, "%s/%I64di.pgm", folderName, timer.printTime());
	else
		sprintf_s(depthFileName, "%s/%s.pgm", folderName, fileName);

	imwrite(depthFileName, irImage);

	SafeRelease(pIrFrame);
}


// Windows function for saving bitmap to file - manually making the header!
HRESULT MyKinectv2::SaveColourBitmapToFile(BYTE* pBitmapBits, LONG lWidth, LONG lHeight, WORD wBitsPerPixel, LPCWSTR lpszFilePath)
{
	DWORD dwByteCount = lWidth * lHeight * (wBitsPerPixel / 8);

	BITMAPINFOHEADER bmpInfoHeader = { 0 };

	bmpInfoHeader.biSize		= sizeof(BITMAPINFOHEADER);		// Size of the header
	bmpInfoHeader.biBitCount	= wBitsPerPixel;				// Bit count
	bmpInfoHeader.biCompression	= BI_RGB;					// Standard RGB, no compression
	bmpInfoHeader.biWidth		= lWidth;						// Width in pixels
	bmpInfoHeader.biHeight		= -lHeight;						// Height in pixels, negative indicates it's stored right-side-up
	bmpInfoHeader.biPlanes		= 1;							// Default
	bmpInfoHeader.biSizeImage	= dwByteCount;					// Image size in bytes
	if (greyFlag)
		bmpInfoHeader.biClrUsed		= 256;

	BITMAPFILEHEADER bfh = { 0 };

	bfh.bfType		= 0x4D42;												// 'M''B', indicates bitmap
	bfh.bfOffBits	= bmpInfoHeader.biSize + sizeof(BITMAPFILEHEADER);		// Offset to the start of pixel data
	bfh.bfSize	= bfh.bfOffBits + bmpInfoHeader.biSizeImage;				// Size of image + headers

	// Create the file on disk to write to
	HANDLE hFile = CreateFileW(lpszFilePath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	// Return if error opening file
	if (NULL == hFile)
	{
		return E_ACCESSDENIED;
	}

	DWORD dwBytesWritten = 0;

	// Write the bitmap file header
	if (!WriteFile(hFile, &bfh, sizeof(bfh), &dwBytesWritten, NULL))
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Write the bitmap info header
	if (!WriteFile(hFile, &bmpInfoHeader, sizeof(bmpInfoHeader), &dwBytesWritten, NULL))
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Write the RGB Data
	if (!WriteFile(hFile, pBitmapBits, bmpInfoHeader.biSizeImage, &dwBytesWritten, NULL))
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Close the file
	CloseHandle(hFile);
	return S_OK;
}


// Used for releasing pointers safely (only important if you have invalid pointers floating around I believe)
template<class Interface>
inline void MyKinectv2::SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}