
#include "MySimulatorv2.h"


// Opens and reads a folder of collected data
// [in]  The root folder containing all the experiment folders
// [in]  The name of the experiment folder to load
// [in]  Wether the folder contains mirrored or normal images. Mirrored is default
void MySimulatorv2::LoadData(char* rootFolder, char* folderToOpen, char obsToLoad_in, ImageMirroring mirroring_in) {

	mirroring = mirroring_in;

	obsToLoad = obsToLoad_in;
	if (obsToLoad_in & OBS_COLOUR_AND_DEPTH) {
		obsToLoad |= OBS_COLOUR;
		obsToLoad |= OBS_DEPTH;
	}

	// Data initialisation
	depthFileCount = 0;
	colourFileCount = 0;

	strcpy(folderName, folderToOpen);		// Keep a copy of the folder name in object memory

	// Check folder exists
	sprintf_s(folderDir, "%s/%s", rootFolder, folderToOpen);
	boost::filesystem::path dir(folderDir);

	if (!(boost::filesystem::exists(folderDir) && boost::filesystem::is_directory(folderDir))) {
		std::cout << "Folder: " << folderToOpen << " does not exist in directory " << rootFolder << std::endl;
		std::cout << "Do you want to continue? [y/n]: ";
		char c;
		std::cin >> c;
		switch (c) {
		case 'y':
			break;
		case 'n':
			exit(EXIT_FAILURE);
			break;
		}
	}

	// If it does then create maps of the included depth and colour files
	if ((obsToLoad & OBS_COLOUR) > 0 || (obsToLoad & OBS_DEPTH) > 0) {
		boost::filesystem::directory_iterator end;
		boost::filesystem::directory_iterator iter(folderDir);
		std::cout << "Loading file list..." << std::endl;
		char format[256], imType, fileExtension[5];
		sprintf(format, "%s/%s\\%%I64d%%c.%%s", rootFolder, folderName);

		while (iter != end) {
			//std::cout << *iter << std::endl;  // Dereference the iterator to access its contents
			std::string filename = (*iter).path().string();
			long long timestamp;
			if (sscanf(filename.c_str(), format, &timestamp, &imType, &fileExtension) == 3) {
				switch (imType) {
				case 'c':
					if (obsToLoad & OBS_COLOUR) {
						colourFileList.insert(std::pair<long long, boost::filesystem::path>(timestamp, (*iter).path()));
						colourFileCount++;
					}
					break;
				case 'd':
					if (obsToLoad & OBS_DEPTH) {
						depthFileList.insert(std::pair<long long, boost::filesystem::path>(timestamp, (*iter).path()));
						depthFileCount++;
					}
					break;
				}
			}

			// Note C++ regex is really really slow!
			//std::tr1::cmatch res;										// Typedef for match results class - const char* type
			//std::tr1::regex rx("\\D+(\\d{6,})([cd])");						// Create the regex - recall the quantifier goes after the target

			//if (std::tr1::regex_search(filename.c_str(), res, rx)) {	// Search for the regex
			//	long long timestamp;
			//	sscanf(res[1].str().c_str(), "%I64d", &timestamp);		// Scanf syntax required for 64-bit integer
			//	std::string type = res[2].str();

			//	if (!type.compare("c"))	{								// Returns ZERO if strings are EQUAL?!?!
			//		colourFileList.insert(std::pair<long long, boost::filesystem::path>(timestamp, (*iter).path()));
			//		colourFileCount++;
			//	}
			//	else if (!type.compare("d")) {
			//		
			//	}
			//}

			iter++;
		}
		std::cout << "Loaded " << colourFileCount << " colour images and " << depthFileCount << " depth images" << std::endl;

		
		if (obsToLoad & OBS_COLOUR) {
			colourIter = colourFileList.begin();
			times.colourTime = colourIter->first;
		}
		else times.colourTime = std::numeric_limits<long long>::max();

		if (obsToLoad & OBS_DEPTH) {
			depthIter = depthFileList.begin();
			times.depthTime = depthIter->first;
		}
		else times.depthTime = std::numeric_limits<long long>::max();
	}
	else {
		times.colourTime = std::numeric_limits<long long>::max();
		times.depthTime = std::numeric_limits<long long>::max();
	}


	// IMMU and GPS data written to single text file
	if (obsToLoad & OBS_IMMU_AND_GPS) {
		sprintf_s(immuGpsFileName, "%s/%s/IMUGPSdata.txt", rootFolder, folderToOpen);
		immuGpsFile.open(immuGpsFileName);
		immuGpsFile.getline(immuGpsDataLine, 256);
		sscanf(immuGpsDataLine, "%*c %I64d", &times.immuGpsTime);
		times.immuGpsTime *= 1000;
		times.immuTime = std::numeric_limits<long long>::max();
		times.gpsTime = std::numeric_limits<long long>::max();
	}
	else {
		times.immuGpsTime = std::numeric_limits<long long>::max();
		
		// Open the gps and immu log text files
		if ((obsToLoad & OBS_IMMU) > 0) {
			sprintf_s(immuFileName, "%s/%s/IMUdata.txt", rootFolder, folderToOpen);
			immuFile.open(immuFileName);
			immuFile.getline(immuDataLine, 256);
			sscanf(immuDataLine, "%*c %I64d", &times.immuTime);
			times.immuTime *= 1000;
		}
		else
			times.immuTime = std::numeric_limits<long long>::max();

		if ((obsToLoad & OBS_GPS) > 0) {
			sprintf_s(gpsFileName, "%s/%s/GPSdata.txt", rootFolder, folderToOpen);
			gpsFile.open(gpsFileName);
			gpsFile.getline(gpsDataLine, 256);
			sscanf(gpsDataLine, "%*c %I64d", &times.gpsTime);
			times.gpsTime *= 1000;
		}
		else
			times.gpsTime = std::numeric_limits<long long>::max();
	}

}



// Destructor closes opened files
MySimulatorv2::~MySimulatorv2() {
	if ((obsToLoad & OBS_IMMU) > 0)
		immuFile.close();
	if ((obsToLoad & OBS_GPS) > 0)
		gpsFile.close();
}


// Sets depth image scaling for greyscale / coloured display
// [in]  Minimum and maximum values to scale the depth values between
//       Anthing outside this range is scaled to 0 or 255 respectively
void MySimulatorv2::SetDepthImageScaling(int mindepth_mm, int maxdepth_mm) {
	minDepthScaling = mindepth_mm;
	maxDepthScaling = maxdepth_mm;
}


// Returns a list of all available files in the experiment folder
// [in]  The type of observation to get the list for
// [ret] An iterator to all the files in the folder
FileIter MySimulatorv2::GetFileList(ObsTypes FILE_TYPE) {
	if (OBS_COLOUR == FILE_TYPE) 
		return colourFileList.begin();
	else if (OBS_DEPTH == FILE_TYPE)
		return depthFileList.begin();
	else {
		std::cout << "Only depth and colour file lists can be returned";
		exit(EXIT_FAILURE);
	}
}


// Plays back data from a single captured stream
// [in]  The stream type to play
void MySimulatorv2::PlayStream(StreamTypes STREAM) {
	std::cout << "Starting stream display. Press q to quit" << std::endl;
	
	if (STREAM <= VISUAL_GREY) {
		PlayVisual(STREAM, colourFileList.begin(), colourFileList.end(), 33);
	}

	else if (STREAM <= DEPTH_GREY_ABS)
		PlayDepth(STREAM, depthFileList.begin(), depthFileList.end(), 33);


	else if (POINTS_GREY <= STREAM ) {
		std::cout << "Functions for visualisation of point clouds as streams currently not written" << std::endl;
		exit(EXIT_FAILURE);
	}
}


// Displays a single image from a stream
// [in]  The stream type to play
// [in]  The image number in the stream to play
void MySimulatorv2::DisplayStreamImage(StreamTypes STREAM, int img_number) {

	if (VISUAL_COLOUR <= STREAM && VISUAL_GREY >= STREAM) {
		if (img_number > colourFileCount) {
			std::cout << "Index exceeds number of colour images" << std::endl;
			exit(EXIT_FAILURE);
		}
		else {
			FileIter iter = colourFileList.begin();
			for (int i = 1; i < img_number; i++, iter++);
			FileIter end = iter;
			end++;
			PlayVisual(STREAM, iter, end, 0);
		}
	}

	// Display the depth image 
	else if (DEPTH_COLOUR_REL <= STREAM) {
		if (img_number > depthFileCount) {
			std::cout << "Index exceeds number of depth images" << std::endl;
			exit(EXIT_FAILURE);
		}
		else {
			FileIter iter = depthFileList.begin();
			for (int i = 1; i < img_number; i++, iter++);
			FileIter end = iter;
			end++;
			if (STREAM >= POINTS_GREY)
				PlayPoints(STREAM, iter, end, 0);
			else 
				PlayDepth(STREAM, iter, end, 0);
		}
	}

}

void MySimulatorv2::LoadColourAndDepthFrame(int d_frame_idx, cv::Mat& cimage_out, cv::Mat& dimage_out) {
	if (d_frame_idx > depthFileCount) {
		std::cout << "Index exceeds number of depth images" << std::endl;
		exit(EXIT_FAILURE);
	}
	else {
		FileIter depthIter = depthFileList.begin();
		for (int i = 1; i < d_frame_idx; i++, depthIter++);
		FileIter colourIter = colourFileList.begin();
		while ((colourIter++)->first < depthIter->first);

		dimage_out = cv::imread(depthIter->second.string(), -1);
		cimage_out = cv::imread(colourIter->second.string(), -1);
	}
}


/*********************************************************************************
*					 S I M U L A T I O N    F U N C T I O N S
*********************************************************************************/
// The HasMoreObs function is used to check whether any of the streams have reached
// their respective end of files, and if not updates an array of timestamps. 
// The type of the minimum timestamp is returned to tell the main function which 
// data retrieval GetNextObs function overload to call. Getting the next observation
// increments internal file pointers but does not read them

// Used to check for the existance of more time-stamped observations within experiment folder
// [ret] A flag indicating the next observation
ObsTypes MySimulatorv2::HasMoreObs(void) {
	
	// Check if any reads have reached the end of the stream
	// Otherwise update the times if we are loading that stream
	if ((obsToLoad & OBS_COLOUR)  && colourFileList.end() == colourIter)
		return OBS_END;
	else if (obsToLoad & OBS_COLOUR) 
		times.colourTime = colourIter->first;

	if ((obsToLoad & OBS_DEPTH)  && depthFileList.end() == depthIter)
		return OBS_END;
	else if (obsToLoad & OBS_DEPTH)
		times.depthTime = depthIter->first;

	if ((obsToLoad & OBS_GPS)  && gpsFile.eof())
		return OBS_END;
	else if (obsToLoad & OBS_GPS) {
		sscanf(gpsDataLine, "%*c %I64d", &times.gpsTime);
		times.gpsTime *= 1000;
	}

	if ((obsToLoad & OBS_IMMU)  && immuFile.eof())
		return OBS_END;
	else if (obsToLoad & OBS_IMMU) {
		sscanf(immuDataLine, "%*c %I64d", &times.immuTime);
		times.immuTime *= 1000;
	}

	if ((obsToLoad & OBS_IMMU_AND_GPS) && immuGpsFile.eof())
		return OBS_END;
	else if (obsToLoad & OBS_IMMU_AND_GPS) {
		sscanf(immuGpsDataLine, "%*c %I64d", &times.immuGpsTime);
		times.immuGpsTime *= 1000;
	}

	// Find minimum in current array of times
	nextObs = times.GetMinType();

	return nextObs;
}


// Gets the next timestamped observation from either colour or depth camera
// [out] The image in which the observation will be stored
// [ret] The timestamp of the observation
long long MySimulatorv2::GetNextObs(cv::Mat& image) {
	cv::Mat tmp;
	long long time = 0;
	if (OBS_DEPTH == nextObs) {
		tmp = cv::imread(depthIter->second.string(),-1);
		if (mirroring)
			cv::flip(tmp, image, 1);
		else
			image = tmp;
		time = times.depthTime;
		depthIter++;
	}
	else if (OBS_COLOUR == nextObs) {
		tmp = cv::imread(colourIter->second.string(), -1);
		if (mirroring)
			cv::flip(tmp, image, 1);
		else
			image = tmp;
		time = times.colourTime;
		colourIter++;
	}
	else {
		std::cout << "Wrong overload called for observation type" << std::endl;
		std::cin.ignore();
	}
	return time;
}


// Gets syncrhonised timestamped observations from both colour and depth cameras
// [out] The colour image in which the observation will be stored
// [out] The depth image in which the observation will be stored
// [ret] The timestamp of the observation
long long MySimulatorv2::GetNextObs(cv::Mat& colour_image_out, cv::Mat& depth_image_out) {
	cv::Mat tmp1, tmp2;
	long long time = 0;
	if (OBS_COLOUR_AND_DEPTH == nextObs) {
		tmp1 = cv::imread(depthIter->second.string(), -1);
		tmp2 = cv::imread(colourIter->second.string(), -1);
		if (mirroring) {
			cv::flip(tmp1, depth_image_out, 1);
			cv::flip(tmp2, colour_image_out, 1);
		}
		else {
			colour_image_out = tmp1;
			depth_image_out = tmp2;
		}
		time = times.colourTime;
		colourIter++;
		depthIter++;
	}
	else {
		std::cout << "Wrong overload called for observation type" << std::endl;
		std::cin.ignore();
	}
	return time;

}

 
// Gets the next timestamped observation from either IMU or GPS
// [out] The string in which the observation will be stored
// [ret] The timestamp of the observation
long long MySimulatorv2::GetNextObs(std::string& line) {
	long long time = 0;
	if (OBS_GPS == nextObs) {
		line = std::string(gpsDataLine);
		time = times.gpsTime;
		gpsFile.getline(gpsDataLine, 256);
	}
	else if (OBS_IMMU == nextObs) {
		line = std::string(immuDataLine);
		time = times.immuTime;
		immuFile.getline(immuDataLine, 256);
	}
	else if (OBS_IMMU_AND_GPS == nextObs) {
		line = std::string(immuGpsDataLine);
		time = times.immuGpsTime;
		immuGpsFile.getline(immuGpsDataLine, 256);
	}
	
	else {
		std::cout << "Wrong overload called for observation type" << std::endl;
		std::cin.ignore();
	}
	return time;
}



/*********************************************************************************
*					   P R I V A T E    F U N C T I O N S
*********************************************************************************/

void MySimulatorv2::PlayDepth(StreamTypes stream, FileIter start, FileIter end, int delay_ms) {
	std::ostringstream windowName;
	windowName << "'" << folderName << "' Depth Image Stream | q to quit | p to pause";
	cv::namedWindow(windowName.str());

	for (FileIter iter = start; iter != end; iter++) {
		
		cv::Mat dimage, tmp = cv::imread(iter->second.string(), -1);
		if (mirroring)
			cv::flip(tmp, dimage, 1);
		else
			dimage = tmp;
		cv::Mat scaledImage, colourImage;
		
		switch (stream) {
		case DEPTH_COLOUR_REL:
			DepthToRelativeIntensity(dimage, scaledImage);
			cv::applyColorMap(scaledImage, colourImage, cv::COLORMAP_JET);
			cv::imshow(windowName.str(), colourImage);
			break;
		case DEPTH_GREY_REL:
			DepthToRelativeIntensity(dimage, scaledImage);
			cv::imshow(windowName.str(), scaledImage);
			break;
		case DEPTH_COLOUR_ABS:
			DepthToAbsoluteIntensity(dimage, scaledImage);
			cv::applyColorMap(scaledImage, colourImage, cv::COLORMAP_JET);
			cv::imshow(windowName.str(), colourImage);
			break;
		case DEPTH_GREY_ABS:
			DepthToAbsoluteIntensity(dimage, scaledImage);
			cv::imshow(windowName.str(), scaledImage);
			break;
		}

		char key = (char)cv::waitKey(delay_ms);
		if (key == 'q' || key == 'Q')
			break;
		if (key == 'p' || key == 'P')
			std::ignore;
	}
	cv::destroyWindow(windowName.str());
}


void MySimulatorv2::PlayVisual(StreamTypes stream, FileIter start, FileIter end, int delay_ms) {
	std::ostringstream windowName;
	windowName << "'" << folderName << "' Colour Image Stream | q to quit | p to pause";
	cv::namedWindow(windowName.str());

	for (FileIter iter = start; iter != end; iter++) {
		
		cv::Mat cimage, tmp = cv::imread(iter->second.string());
		if (mirroring)
			cv::flip(tmp, cimage, 1);
		else
			cimage = tmp;
		cv::Mat gimage;

		if (VISUAL_COLOUR == stream)
			cv::imshow(windowName.str(), cimage);
		else if (VISUAL_GREY == stream && cimage.channels() > 1) {
			cv::cvtColor(cimage, gimage, CV_RGB2GRAY);
			imshow(windowName.str(), gimage);
		}
		else if (VISUAL_GREY == stream && cimage.channels() == 1) 
			cv::imshow(windowName.str(), cimage);
		else if (VISUAL_COLOUR == stream && cimage.channels() == 1) {
			std::cout << "Cannot display greyscale image in colour. Press any key to exit." << std::endl;
			std::cin.ignore();
			break;
		}

		char key = (char)cv::waitKey(delay_ms);
		if (key == 'q' || key == 'Q')
			break;
		if (key == 'p' || key == 'P')
			std::ignore;
	}

	cv::destroyWindow(windowName.str());
}



void MySimulatorv2::PlayPoints(StreamTypes stream, FileIter start, FileIter end, int delay_ms) {

	std::ostringstream windowName;
	windowName << "'" << folderName << "' Point Cloud | q to quit | p to pause";
	FileIter iter = start;

	for (FileIter iter = start; iter != end; iter++) {
		cv::Mat dimage, tmp = cv::imread(iter->second.string(), -1);
		if (mirroring)
			cv::flip(tmp, dimage, 1);
		else
			dimage = tmp;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		KinectCamParams cam;
		ProjectDepthImageToCloud(dimage, cloud, cam.depth_fx, cam.depth_fy, cam.depth_cx, cam.depth_cy);

		if (stream == POINTS_COLOUR) {
			long long depth_time = (*start).first;
			long long difference = 1e10;
			boost::filesystem::path closest_match;		// Path to matching colour file
			// Some code checking either side of the depth image for the colour image
			
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr colourcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			// Function associating a colour image and depth point cloud. 
		}

		pcl::visualization::CloudViewer viewer(windowName.str());
		viewer.showCloud(cloud);
		while (!viewer.wasStopped()) {}
	} 
}



// Scales a 16bit depth map to 8bit intensity map between the minimum and maximum ranges in the image
void MySimulatorv2::DepthToRelativeIntensity(cv::Mat& depthRaw, cv::Mat& depthImOut) {
	
	// Find minimum and maximum values of the depth map
	unsigned short  mindepth = 10000, maxdepth = 0;
	for (int row = 0; row < depthRaw.rows; row++) {
		for (int col = 0; col < depthRaw.cols; col++) {
			unsigned short depth = depthRaw.at<unsigned short>(row, col);
			if (depth < mindepth && depth != 0)
				mindepth = depth;
			if (depth > maxdepth)
				maxdepth = depth;
		}
	}

	// Scale all values in output map accordingly
	depthImOut.create(depthRaw.rows, depthRaw.cols, CV_8UC1);
	for (int row = 0; row < depthRaw.rows; row++) {
		for (int col = 0; col < depthRaw.cols; col++) {
			unsigned short depth = depthRaw.at<unsigned short>(row, col);
			if (depth >= mindepth && depth <= maxdepth) {
				unsigned char depthValue = 255 * (float)(depth - mindepth) / (maxdepth - mindepth);
				depthImOut.at<unsigned char>(row, col) = depthValue;
			}
			else if (depth >= maxdepth)
				depthImOut.at<unsigned char>(row, col) = 255;
			else 
				depthImOut.at<unsigned char>(row, col) = 0;
		}
	}
}

// Scales a 16bit depth map to 8bit intensity map between pre-specified range boundaries
void MySimulatorv2::DepthToAbsoluteIntensity(cv::Mat& depthRaw, cv::Mat& depthImOut) {
	depthImOut.create(depthRaw.rows, depthRaw.cols, CV_8UC1);
	for (int row = 0; row < depthRaw.rows; row++) {
		for (int col = 0; col < depthRaw.cols; col++) {
			unsigned short depth = depthRaw.at<unsigned short>(row, col);
			if (depth >= minDepthScaling && depth <= maxDepthScaling) {
				unsigned char depthValue = 255 * (float)(depth - minDepthScaling) / (maxDepthScaling - minDepthScaling);
				depthImOut.at<unsigned char>(row, col) = depthValue;
			}
			else if (depth >= maxDepthScaling)
				depthImOut.at<unsigned char>(row, col) = 255;
			else
				depthImOut.at<unsigned char>(row, col) = 0;
		}
	}
}



// Return a stream image using an index
void MySimulatorv2::LoadStreamCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int img_number) {

	if (img_number > depthFileCount) {
		std::cout << "Index exceeds number of point clouds" << std::endl;
		exit(EXIT_FAILURE);
	}
	else {
		std::multimap<long long, boost::filesystem::path>::iterator iter = depthFileList.begin();
		for (int i = 1; i < img_number; i++, iter++);

		cv::Mat dimage, tmp = cv::imread(iter->second.string(), -1);
		if (mirroring)
			cv::flip(tmp, dimage, 1);
		else
			dimage = tmp;
		//ProjectDepthImageToCloud(dimage, cloud);
	}

}


// Note, possibly use this function to add text to the image
// to display which image number it is
// std::ostringstream imageName;
// imageName.clear();
// int img_number;
// imageName << "Image: " << img_number++ ;
// cv::Mat cimage = cv::imread(iter->second.string()); 
// cv::putText(cimage, imageName.string(), cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250));
// cv::namedWindow("Colour Image Stream | q to quit");
// cv::imshow("Colour Image Stream | q to quit", cimage);

//// The following is my original code for terminating the observation reading - it waits until all data sources are finished, not just the first one to finish
//if (obsEndFlags[OBS_COLOUR] && obsEndFlags[OBS_DEPTH] && obsEndFlags[OBS_IMMU] && obsEndFlags[OBS_GPS])
//return OBS_END;
//else if (OBS_COLOUR == min_idx) {
//	if (colourFileList.end() == colourIter) {
//		obsEndFlags[OBS_COLOUR] = true;
//		times[OBS_COLOUR] = std::numeric_limits<long long>::max();
//	}
//	else {
//		colourIter++;
//		times[OBS_COLOUR] = colourIter->first;
//	}
//	nextObs = OBS_COLOUR;
//}
//else if (OBS_DEPTH == min_idx) {
//	if (depthFileList.end() == depthIter) {
//		obsEndFlags[OBS_DEPTH] = true;
//		times[OBS_DEPTH] = std::numeric_limits<long long>::max();
//	}
//	else {
//		depthIter++;
//		times[OBS_DEPTH] = depthIter->first;
//	}
//	nextObs = OBS_DEPTH;
//}
//else if (OBS_GPS == min_idx) {
//	if (gpsFile.eof()) {
//		obsEndFlags[OBS_GPS] = true;
//		times[OBS_GPS] = std::numeric_limits<long long>::max();
//	}
//	else {
//		gpsFile.getline(gpsDataLine, 256);
//		sscanf(gpsDataLine, "%*c %I64d", &gpsTime);
//		times[OBS_GPS] = 1000 * gpsTime;
//	}
//	nextObs = OBS_GPS;
//}
//else if (OBS_IMMU == min_idx) {
//	if (immuFile.eof()) {
//		obsEndFlags[OBS_IMMU] = true;
//		times[OBS_IMMU] = std::numeric_limits<long long>::max();
//	}
//	else {
//		immuFile.getline(immuDataLine, 256);
//		sscanf(immuDataLine, "%*c %I64d", &immuTime);
//		times[OBS_IMMU] = 1000 * immuTime;
//	}
//	nextObs = OBS_IMMU;
//}