/***************************************************************************************

File Name:		MyKinectv2.h
Author:			Alex Bunting
Date Modified:  24/7/14

Description:
Contains the definition of the MyKinectv2 class, used to interface with the Kinect for 
Windowsv2 hardware. Contains initialisation functions and functions for capturing 
synchronised data streams as well as individual images. Also includes struct 
KinectCamParams which holds Kinectv2 intrinsic and extrinsic sensor calibration data, 
typically required for coordinate transforms. 

TODO: 
- Add ability to save single named images with synchronous viewer
- Add viewers for other stream types
- Add screenshot option for other stream types
- Add option to switch viewers while running
- Work out conditions for SaveImage (synced) to return true
- Add option for relative depth stream viewing?

****************************************************************************************/

#ifndef MY_KINECTV2_H
#define MY_KINECTV2_H


/****  DEFINES  ****/
#undef max
#undef min


/****  MY INCLUDES  ****/
#include "MyTimer.h"


/****  EXTERNAL INCLUDES  ****/
#include "Kinect.h"

#include <boost/filesystem.hpp>

#include <iostream>
#include <conio.h>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\contrib\contrib.hpp>

// These need to be immediately before any PCL includes
// To remove windows defines of min and max
#undef max
#undef min
#include <pcl/common/eigen.h>
#include <pcl/common/impl/eigen.hpp>


/****  NAMESPACESS  ****/
using cv::Mat;


struct KinectCamParams {
public:
	float colour_fx, colour_fy, colour_cx, colour_cy;
	float depth_fx, depth_fy, depth_cx, depth_cy;
	static const int    cColorWidth = 1920;
	static const int    cColorHeight = 1080;
	static const int    cDepthWidth = 512;
	static const int    cDepthHeight = 424;
	Eigen::Matrix4f depthToColourTransform;
	
	// Hard-coded default values for kinect camera parameters
	// Primarily used by coordinate mapping modules
	// Found with Bouget's toolbox for stereo camera calibration
	KinectCamParams() {
		depth_fx = 361.41463;
		depth_fy = 361.13281;
		depth_cx = 250.92297;
		depth_cy = 203.69426;
		colour_fx = 1064.00189;
		colour_fy = 1063.74614;
		colour_cx = 946.74256;
		colour_cy = 539.82995;
		depthToColourTransform <<
			0.9998, 0.0176, 0.0080, 0.0522,
			-0.0175, 0.9997, -0.0154, -0.0008,
			-0.0082, 0.0153, 0.9998, 0.0059,
			0.0000, 0.0000, 0.0000, 1.0000;
	}

};

typedef enum {
	ASYNC_CAP	= 1,
	SYNC_CAP	= 2,
} CaptureTypes;


class MyKinectv2 {
	
public:
	// Constructor for Kinectv2 object
	// Allocates memory for image buffers
	MyKinectv2();


	// Destructor for Kinectv2 object 
	// Performs memroy deallocation for image buffers
	~MyKinectv2();


	// Connects to the default Kinectv2 sensor for synchronised frame capture
	// [in]  The folder in which data from the kinect should be saved
	// [in]  Which streams to capture data from with the Update() command
	// [in]  Whether colour images should be greyscale. Reduces data size by 1/4 but may slow capture
	void InitSyncCapture(char* folderDirectory, int sourceTypes, bool greyscale = false);


	// Connects to default Kinectv2 sensor for asynchronous frame capture
	// [in]  The folder in which data from the kinect should be saved
	// [in]  Which streams to capture data from with the Update() command
	// [in]  Whether colour images should be greyscale. Reduces data size by 1/4 but may slow capture
	void InitAsyncCapture(char* folderDirectory, int sourceTypes, bool greyscale = false);

	
	// Quick function for capturing synchronised timestamped colour and depth data
	// Captures data at max rate until 'q' is pressed to exit
	void QuickSyncCapture(void);


	// Captures depth and colour data asynchronously at maximum rate
	// Captures data at max rate until 'q' is pressed to exit
	void QuickAsyncCapture(void);


	// Starts a viewer for the frame source 
	// [in]  The frame source type to start viewing
	void StartViewer(int sourceTypes_in, int depth_min_distance = 500, int depth_max_distance = 4500);


	// Poll this function to capture data from the initialised streams
	// [ret] 0 if no data available, or 1 if any data captured
	//       No guarantee that all frame types initialised will be captured each update
	int GrabFrames(void);


	// Saves a single colour image of the given name
	// [in]  The name of the image to save
	//       Use NULL for a timestamp to be used as the name
	void SaveColourImage(const char* imageName = NULL);


	// Saves a single depth image of the given file name
	// [in]  The name of the image to save
	//       Use NULL for a timestamp to be used as the name
	void SaveDepthImage(const char* imageName = NULL);


	// Saves a single IR image of the given file name
	// [in]  The name of the image to save
	//       Use NULL for a timestamp to be used as the name
	void SaveIRImage(const char* imageName = NULL);


	// Saves a single image of the given types and file name
	// [in]  The stream source to save. Requires previous InitXXXXCapture call
	// [in]  The name of the image to save
	//       Use NULL for a timestamp to be used as the name appended with image type ('i', 'c', or 'd')
	//       Colour images are saved as raw 8-bit RGB name.bmp
	//       Depth images are saved raw as 16-bit greyscale name.pgm
	//       IR images are scaled between min and max values as 8-bit name.png
	// [ret] True if sync capture type and the frames were received from a single 
	//       MultiFrame object. False if multiple multi-frames were required to get 
	//       all images and therefore not truely synchronised
	bool SaveImage(int frameType, const char* imageName = NULL);


	// Saves synchronised depth and colour images of a given file name
	// [in]  The name of the images to save - one will be name.bmp, the other name.pgm
	//       Use NULL for a timestamp to be used as the name
	void SaveDepthAndColourImage(const char* imageName = NULL);


	// Allows a different set of depth intrinsic parameters to be associated with the object
	// [in]  The intrinsics of focal length and camera centre in pixels
	void SetDepthIntrinsics(float& fx, float& fy, float& cx, float& cy);


	// Allows a different set of colour intrinsic parameters to be associated with the object
	// [in]  The intrinsics of focal length and camera centre in pixels
	void SetColourIntrinsics(float& fx, float& fy, float& cx, float& cy);


	// Allows a different set of extrinsic parameters between the two sensors to be associated with the object
	// [in]  The homogenous transformation matrix between the two sensors in mm
	void SetSensorRelativeTransform(Eigen::Matrix4f relativeTransform);


	// Returns the camera parameters of the current object
	// [ret] The KinectCamParams struct of current parameters
	KinectCamParams GetCameraParameters(void);


	// Definitions for the sizes of images
	static const int        cColorWidth = 1920;
	static const int        cColorHeight = 1080;

	static const int        cDepthWidth = 512;
	static const int        cDepthHeight = 424;

private:
	// Pointer to current Kinect
	IKinectSensor*				kinect;		

	// For storing what frames we are capturing and how
	CaptureTypes capType;
	int frameTypes;

	// For synchronised frame reception
	IMultiSourceFrameReader*	multiReader;
	IMultiSourceFrame*			multiFrame;

	// For colour and depth frames
	IColorFrameReader*			colourReader;
	IDepthFrameReader*			depthReader;
	IInfraredFrameReader*		irReader;
	IColorFrameReference*		colourRef;
	IDepthFrameReference*		depthRef; 
	IInfraredFrameReference*	irRef;
	IColorFrame*				colourFrame;
	IDepthFrame*				depthFrame;
	IInfraredFrame*				irFrame;

	// Pointers to buffers which store image data
	RGBQUAD*					m_pColorRGBX;
	RGBQUAD*					m_pDepthRGBX;
	RGBQUAD*					m_pIrRGBX;
	BYTE*						m_pColourGreyscale;

	// Used for finding coordinate transforms?
	ICoordinateMapper*			coordMapper;

	// For saving timestamps and finding fps
	Timer timer;
	bool timingFlag;
	bool greyFlag;

	// Strings for naming files
	char colourFileName[256];
	char depthFileName[256];
	char irFileName[256];
	char folderName[128];

	// Stores intrinsic and extrinsic camera parameters
	KinectCamParams camParams;

	// Draws a depth frame scaled between min and max for 'jet' colourmap
	void DrawDepth(IDepthFrame* pDepthFrame_in, cv::Mat& depthImage_out, int min_depth = 500, int max_depth = 4500);

	// Draws a colour frame using OpenCV

	// Draws an IR frame using OpenCV

	// Saves a colour frame to file
	void SaveColour(IColorFrame* pColorFrame, const char* fileName = NULL);

	// Saves a depth frame to file (raw 16 bit pgm)
	void SaveDepth(IDepthFrame* pDepthFrame, const char* fileName = NULL);

	// Saves an IR frame to file (8 most significant bits or 16bit value)
	void SaveIR(IInfraredFrame* pIrFrame, const char* fileName = NULL);

	// Converts a Kinectv2 depth frame to a cv::Mat USHORT structure
	void FrameToImageD(IDepthFrame* pDepthFrame, Mat& depthImage);

	// Converts a Kinectv2 infrared frame to a cv::Mat USHORT structure
	void FrameToImageI(IInfraredFrame* pIrFrame, Mat& irImage);

	// Windows-based function for saving a bitmap to file, might replace with OpenCV equivalent
	HRESULT SaveColourBitmapToFile(BYTE* pBitmapBits, LONG lWidth, LONG lHeight, WORD wBitsPerPixel, LPCWSTR lpszFilePath);

	// For releasing pointers safely
	template<class Interface> inline void SafeRelease(Interface *& pInterfaceToRelease);
};

#endif