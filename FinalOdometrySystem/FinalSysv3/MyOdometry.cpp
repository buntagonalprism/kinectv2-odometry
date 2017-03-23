#include "MyOdometry.h"

MyOdometry::MyOdometry(int numframes_in) : numframes(numframes_in) {

	kps.resize(numframes);
	kp_desc.resize(numframes);
	mimages.resize(numframes);
	gimages.resize(numframes);
	dimages.resize(numframes);
	kp_clouds.resize(numframes);
	cam_poses.resize(numframes);
	ba_poses.resize(numframes);

	concat_VO_pose = Eigen::Matrix4f::Identity();
	relative_VO_pose = Eigen::Matrix4f::Identity();
	last_ba_pose = Eigen::Matrix4f::Identity();
	cam_poses[0] = Eigen::Matrix4f::Identity();

	num_meas = 0;
	frame = 0;
	ahrs_cnt = 0;


}

MyOdometry::~MyOdometry() {
	rgbdPosesFile.close();
	sbaPosesFile.close();
	poseTimesFile.close();
	ahrsFile.close();
	ahrsTimes.close();
	rawImmuOutFile.close();
	calibImmuOutFile.close();
	keypointCountsFiles.close();
	icpfForwardPosesFile.close(); 
	icpReversePosesFile.close();
	depthImStatsFile.close();
}

void MyOdometry::InitDataSource(char* inputFolderDir, char* inputFolderName, int obsToLoad, ImageMirroring mirroring) {

	sim.LoadData(inputFolderDir, inputFolderName, obsToLoad, mirroring);

}


void MyOdometry::InitOutput(char* outputFolderDir, char* outputFolderName) {

	VerifyOutputDir(outputFolderDir, outputFolderName);

	OpenOutputFiles(rgbdPosesFile, sbaPosesFile, poseTimesFile, ahrsFile, ahrsTimes, rawImmuOutFile, 
		calibImmuOutFile, keypointCountsFiles, icpfForwardPosesFile, icpReversePosesFile, depthImStatsFile,
		outputFolderDir, outputFolderName);

}

void MyOdometry::InitStarTracker(int maxSize, int responseThresh, int lineThreshProjected, int lineThresBinarised, int suppressNonmaxSize) {
	starOpts.lineThresBinarised = lineThresBinarised;
	starOpts.maxSize = maxSize;
	starOpts.lineThreshProjected = lineThreshProjected;
	starOpts.suppressNonmaxSize = suppressNonmaxSize;
	starOpts.responseThresh = responseThresh;
	featureDetector = cv::StarFeatureDetector(maxSize, responseThresh, lineThreshProjected, lineThresBinarised, suppressNonmaxSize);

}

void MyOdometry::InitBriefDescriptor(int bytes) {

	featureExtractor = cv::BriefDescriptorExtractor(bytes);
}

void MyOdometry::InitMatcher(int matchType, float best_ratio) {

}


void MyOdometry::InitSBA(void) {
	sbaParams.numframes = numframes;
	
	// Number of parameters per model type
	sbaParams.cnp = 6;
	sbaParams.mnp = 2;
	sbaParams.pnp = 3;

	sbaParams.globs.cnp = sbaParams.cnp;
	sbaParams.globs.pnp = sbaParams.pnp;
	sbaParams.globs.mnp = sbaParams.mnp;

	// Camera parameters
	KinectCamParams cam;
	sbaParams.globs.intrcalib[0] = cam.colour_fx;					// fu
	sbaParams.globs.intrcalib[1] = cam.colour_cx;					// u0
	sbaParams.globs.intrcalib[2] = cam.colour_cy;;					// v0
	sbaParams.globs.intrcalib[3] = cam.colour_fy / cam.colour_fx;	// ar = fv/fu
	sbaParams.globs.intrcalib[4] = 0.0;								// skew

	sbaParams.globs.ptparams = NULL;
	sbaParams.globs.camparams = NULL;

	// SBA Options
	sbaParams.opts[0] = SBA_INIT_MU;
	sbaParams.opts[1] = SBA_STOP_THRESH;
	sbaParams.opts[2] = SBA_STOP_THRESH;
	sbaParams.opts[3] = SBA_STOP_THRESH;
	//opts[3]=0.05*numprojs;	// uncomment to force termination if the average reprojection error drops below 0.05
	sbaParams.opts[4] = 0.0;
	//opts[4]=1E-05;	// uncomment to force termination if the relative reduction in the RMS reprojection error drops below 1E-05

	sbaParams.verbose = 1; // Increasing this increases output data including initial error and mean pixel reprojection error

}

void MyOdometry::InitRansac(int min_sample_size_in, int ransac_iterations_in, double inlier_thresh_in, KinectCamParams cam_in){

	ranPams = MyRGBD::RansacParams(min_sample_size_in, ransac_iterations_in, inlier_thresh_in, cam_in);

}

void MyOdometry::StartExperimental() {
	
	KinectCamParams cam;

	float* camArray = new float[8 + 16];

	camArray[CCX] = cam.colour_cx;
	camArray[CCY] = cam.colour_cy;
	camArray[CFX] = cam.colour_fx;
	camArray[CFY] = cam.colour_fy;
	camArray[DCX] = cam.depth_cx;
	camArray[DCY] = cam.depth_cy;
	camArray[DFX] = cam.depth_fx;
	camArray[DFY] = cam.depth_fy;
	Eigen::Matrix4f poseDepthWrtColour = cam.depthToColourTransform.inverse();
	memcpy(&camArray[TRANS], poseDepthWrtColour.data(), 16 * sizeof(float));

	int* imArray = new int[4];

	imArray[CNX] = cam.cColorWidth;
	imArray[CNY] = cam.cColorHeight;
	imArray[DNX] = cam.cDepthWidth;
	imArray[DNY] = cam.cDepthHeight; 
	
	//pcl::gpu::KinfuTracker kinfu(cam.cDepthHeight, cam.cDepthWidth);
	//kinfu.setDepthIntrinsics(cam.depth_fx, cam.depth_fy, cam.depth_cx, cam.depth_fy);
	//Eigen::Affine3f concat_ICP_pose = Eigen::Affine3f::Identity();
	//kinfu.setInitalCameraPose(concat_ICP_pose);

	//sim.SetDepthImageScaling(500, 4500);
	//sim.PlayStream(DEPTH_COLOUR_ABS);
	//sim.PlayStream(VISUAL_GREY);

	//ranPams.ransac_iterations = 400;
	ranPams.inlier_thresh_sqd = 9;

	float imStats[192];

	ObsTypes type;
	long long time, newtime = 0, oldtime = 0;
	bool drawDebug = false;
	bool drawTracks = false;

	while ((type = sim.HasMoreObs()) != OBS_END) {
		switch (type) {
		case OBS_IMMU_AND_GPS:
			time = sim.GetNextObs(immudataline);

			if (ahrs.ScanImmuLine(immudataline, timestamp, accel, gyro, mag)) {

				if (ahrs_cnt++ == 0)
					ahrs.init(accel, mag, time);
				else {
					ahrs.estimate(accel, gyro, mag, time, rotation);

					ahrs.WriteRotation(ahrsFile, rotation);

					ahrsTimes << time << std::endl;

					ahrs.WriteSensors(rawImmuOutFile, time, accel, gyro, mag);

					ahrs.GetCalibSensors(accel, gyro, mag);

					ahrs.WriteSensors(calibImmuOutFile, time, accel, gyro, mag);

				}
			}
			break;
		case OBS_COLOUR:
			sim.GetNextObs(dummy);
			break;
		case OBS_DEPTH:
			sim.GetNextObs(dummy);
			break;
		case OBS_COLOUR_AND_DEPTH:
			oldtime = newtime;
			newtime = sim.GetNextObs(cimage, dimages[frame]);
			if (cimage.channels() == 3)
				cv::cvtColor(cimage, gimages[frame], CV_RGB2GRAY);
			else
				cimage.copyTo(gimages[frame]);


			if (newtime - oldtime > 500000) {
				// Tracking likely slipped and we need to restart from this image
				world_pts.clear();
				num_meas = 0;
				frame = 0;
			}

			//timer.tic();
			//MyKinectv2Utils::MapKinectv2DepthToColour(dimage, mimages[frame]);
			//timer.toc("Depth image mapped to colour space");

	
			timer.tic();
			MyCudaUtils::MapDepthToColour(dimages[frame], mimages[frame], camArray, imArray);
			timer.toc("Parallel depth image mapped");
			
			kps[frame].keypoints.clear();
			bool rgbdFailure = false;
			int thresh = starOpts.responseThresh;
			while (kps[frame].keypoints.size() < 3 && thresh > 10) {
				timer.tic();
				MyCudaUtils::ComputeStarKeypoints(gimages[frame], kps[frame].keypoints, starOpts.maxSize, starOpts.responseThresh, starOpts.lineThreshProjected, starOpts.lineThresBinarised, starOpts.suppressNonmaxSize);
				timer.toc("Star keypoints found");
				thresh -= 5;
			}
			if (thresh == 10) 
				rgbdFailure = true;
	
			if (odoOpts.drawKps) MyDrawingUtils::DrawKeypoints(gimages[frame], kps[frame].keypoints);


			kps[frame].init_idxs();

	
			timer.tic();
			MyCudaUtils::ComputeBriefDescriptors(gimages[frame], kps[frame].keypoints, kp_desc[frame]);
			timer.toc("Brief descriptors computed");
	

			if (frame > 0) {
			
				timer.tic();
				MyCudaUtils::BruteHammingCUDA(kp_desc[frame], kp_desc[frame - 1], match_idx, match_dist);
				timer.toc("Keypoints brute-force matched");
		
				timer.tic();
				good_matches = MyRGBD::LowesFilterFeatures(match_idx, match_dist, kps[frame].keypoints, kps[frame - 1].keypoints, gimages[frame], gimages[frame - 1]);
				timer.toc("Matches filtered with Lowe's Algo");
				
				if (drawDebug) MyDrawingUtils::DrawMatches(gimages, kps, good_matches, frame);

				timer.tic();
				projected_matches = MyRGBD::ProjectKeypointsToCloud(good_matches, kp_clouds[frame], kp_clouds[frame - 1], mimages[frame], mimages[frame - 1], kps[frame].keypoints, kps[frame - 1].keypoints);
				timer.toc("Projected matches to cloud");
		       
				keypointCountsFiles << kps[frame].keypoints.size() << " " << good_matches.size() << " " << projected_matches.size();

				if (odoOpts.drawMatches) MyDrawingUtils::DrawMatches(gimages, kps, inlier_matches, frame);

				
				if (projected_matches.size() > 20 && !rgbdFailure) {
					timer.tic();
					inlier_matches = MyCudaOdometry::RansacFilterFeatures(projected_matches, kps, kp_clouds, frame, ranPams, camArray);
					timer.toc("GPU Ransac filtering complete");

					keypointCountsFiles << " " << inlier_matches.size() << std::endl;

					timer.tic();
					MyRGBD::EstimateTransform(kp_clouds[frame], kp_clouds[frame - 1], relative_VO_pose);
					timer.toc("Estimating transform");

					rgbdPosesFile << relative_VO_pose << std::endl;

					concat_VO_pose = concat_VO_pose*relative_VO_pose;

					cam_poses[frame] = concat_VO_pose;

					timer.tic();
					MySBA::AddPointsAndMeasurements(world_pts, kps, inlier_matches, kp_clouds, cam_poses, num_meas, frame, numframes);
					timer.toc("Adding data to SBA arrays");
				}
				else {
					keypointCountsFiles << " " << 0 << std::endl;

					rgbdPosesFile << Eigen::Matrix4f::Identity() << std::endl;
					
					rgbdFailure = true;
				}

				poseTimesFile << newtime << std::endl;

				// Use the RGBD estimate to initialise the ICP
				Eigen::Matrix4f forwardEst = relative_VO_pose.inverse();
				Eigen::Matrix4f reverseEst = relative_VO_pose;

				
				//cv::Mat scaleup_new, scaleup_old;
				//scaleup_new = dimages[frame] * 10;
				//scaleup_old = dimages[frame-1] * 10;
				//cv::imshow("New image", scaleup_new);
				//while (cv::waitKey(1) != 'q');
				//cv::imshow("Old image", scaleup_old);
				//while (cv::waitKey(1) != 'q');
				//cv::destroyAllWindows();
				

				timer.tic();
				MyCudaUtils::DenseICP(dimages[frame], dimages[frame - 1], camArray, imArray, forwardEst.data(), imStats);
				
				for (int i = 0; i < 192; i++)
					depthImStatsFile << imStats[i] << " ";
				depthImStatsFile << std::endl;

				MyCudaUtils::DenseICP(dimages[frame - 1], dimages[frame], camArray, imArray, reverseEst.data(), imStats);
				timer.toc("Forward and reverse ICP");

				icpfForwardPosesFile << forwardEst << std::endl;

				icpReversePosesFile << reverseEst << std::endl;

				// Tidy up if tracking failed, use ICP for next round
				if (rgbdFailure) {
					relative_VO_pose = forwardEst;
					frame = 0;
					world_pts.clear();
					num_meas = 0;
					break;
				}
				
			}
 			frame++;
			
			if (frame == numframes) {

				if (odoOpts.drawTraces) MyDrawingUtils::DrawKeypointTracks(gimages[frame - 1], world_pts, numframes, 0, 10);

				timer.tic();

				MySBA::BundleAdjust(world_pts, cam_poses, ba_poses, num_meas, sbaParams);

				std::cout << time << std::endl;

				MySBA::WriteRelativePoses(std::cout, ba_poses);

				MySBA::WriteRelativePoses(sbaPosesFile, ba_poses);

				// Free the first object in the vector

				// Set the last frame of current set as first frame of next set by copying pointers
				kp_desc[0] = kp_desc[numframes - 1];  		// Copy Mat header info
				mimages[0] = mimages[numframes - 1];
				gimages[0] = gimages[numframes - 1];
				dimages[0] = dimages[numframes - 1];
				kp_clouds[0] = kp_clouds[numframes - 1]; 
				kps[0] = kps[numframes - 1];
				kps[0].init_idxs();
			
				// Clear the set of 3D points and the initial rotations
				world_pts.clear();			// Clear the vector of pointers
				num_meas = 0;

				// Set camera poses to start again
				cam_poses[0] = Eigen::Matrix4f::Identity();
				concat_VO_pose = cam_poses[0];

				frame = 1;

				timer.toc("Bundle adjustment performed");
			}
			

			
			

		}
	}

	delete[] imArray;
	delete[] camArray;
}

void MyOdometry::Start(int framesToProcess) {
	ObsTypes type;
	long long time;
	bool printOutput = false;
	int framesProcessed = 0;

	if (framesToProcess < 0)
		framesToProcess = std::numeric_limits<int>::max();


	while ((type = sim.HasMoreObs())!= OBS_END  && framesProcessed < framesToProcess) {
		switch (type) {
		case OBS_IMMU_AND_GPS:

			time = sim.GetNextObs(immudataline);

			if (ahrs.ScanImmuLine(immudataline, timestamp, accel, gyro, mag)) {

				if (ahrs_cnt++ == 0)
					ahrs.init(accel, mag, time);
				else {
					ahrs.estimate(accel, gyro, mag, time, rotation);

					ahrs.WriteRotation(ahrsFile, rotation);
					
					ahrsTimes << time << std::endl;

					ahrs.WriteSensors(rawImmuOutFile, time, accel, gyro, mag);
					
					ahrs.GetCalibSensors(accel, gyro, mag);

					ahrs.WriteSensors(calibImmuOutFile, time, accel, gyro, mag);
					
				}
			}
			break;

		case OBS_COLOUR:
			sim.GetNextObs(dummy);
			break;
		case OBS_DEPTH:
			sim.GetNextObs(dummy);
			break;
		case OBS_COLOUR_AND_DEPTH:
			time = sim.GetNextObs(cimage, dimage);

			timer.tic();
			MyKinectv2Utils::MapKinectv2DepthToColour(dimage, mimages[frame]);
			timer.toc("Serial mapping from depth to colour space");

			//MyCudaUtils::MapColourToDepth(cimage, dimage, dummy);

			if (cimage.channels() > 1)
				cv::cvtColor(cimage, gimages[frame], CV_BGR2GRAY);
			else
				gimages[frame] = cimage;


			
			Timer timerlvl2;

			//timerlvl2.tic(odoOpts.timeFeatures);
			//timer.tic(odoOpts.timeDetector);
			timer.tic();
			featureDetector.detect(gimages[frame], kps[frame].keypoints);
			timer.toc("Serial Star Keypoints found");
			//timer.toc("Keypoints found with STAR", odoOpts.timeDetector);

			//timer.tic(odoOpts.timeExtractor);
			timer.tic();
			featureExtractor.compute(gimages[frame], kps[frame].keypoints, kp_desc[frame]);
			timer.toc("Serial BRIEF descriptors computed");
			//timer.toc("Descriptors computed with BRIEF", odoOpts.timeExtractor);
			//timerlvl2.toc("Keypoints found and described", odoOpts.timeFeatures);

			if (odoOpts.drawKps) MyDrawingUtils::DrawKeypoints(gimages[frame], kps[frame].keypoints);

			kps[frame].init_idxs();

			if (frame == 0) {

				cam_poses[frame] = last_ba_pose;
				num_meas = 0;

			}
			else {
				// So each one of these functions needs a timestamp thing added to it
				timer.tic();
				MyRGBD::MatchFeatures(kp_desc[frame], kp_desc[frame - 1], match_idx, match_dist);
				timer.toc("Serial feature matching");

				good_matches = MyRGBD::LowesFilterFeatures(match_idx, match_dist, kps[frame].keypoints, kps[frame - 1].keypoints, gimages[frame], gimages[frame - 1]);
					
				//MyDrawingUtils::DrawMatches(gimages, kps, good_matches, frame);

				projected_matches = MyRGBD::ProjectKeypointsToCloud(good_matches, kp_clouds[frame], kp_clouds[frame - 1], mimages[frame], mimages[frame - 1], kps[frame].keypoints, kps[frame - 1].keypoints);

				timer.tic();
				inlier_matches = MyRGBD::RansacFilterFeatures(projected_matches, kps[frame].keypoints, kps[frame - 1].keypoints, kp_clouds[frame], kp_clouds[frame - 1], ranPams);
				timer.toc("Serial RANSAC");

				if (odoOpts.drawMatches) MyDrawingUtils::DrawMatches(gimages, kps, inlier_matches, frame);

				// TODO: Work out the order this works. Calling it this way gets positive results like expected
				MyRGBD::EstimateTransform(kp_clouds[frame], kp_clouds[frame - 1], relative_VO_pose);

				rgbdPosesFile << relative_VO_pose << std::endl;
				 
				poseTimesFile << time << std::endl;

				concat_VO_pose = concat_VO_pose*relative_VO_pose;

				cam_poses[frame] = concat_VO_pose;

				MySBA::AddPointsAndMeasurements(world_pts, kps, inlier_matches, kp_clouds, cam_poses, num_meas, frame, numframes);

			}
			frame++;
			framesProcessed++;

			if (frame == numframes) {

				MySBA::BundleAdjust(world_pts, cam_poses, ba_poses, num_meas, sbaParams);

				std::cout << time << std::endl;

				MySBA::WriteRelativePoses(std::cout, ba_poses);

				MySBA::WriteRelativePoses(sbaPosesFile, ba_poses);


				// Copy last element to new first element
				kp_desc[0] = kp_desc[numframes - 1];  		// Copy Mat header info
				mimages[0] = mimages[numframes - 1];
				gimages[0] = gimages[numframes - 1];
				kp_clouds[0] = kp_clouds[numframes - 1]; 
				kps[0] = kps[numframes - 1]; 
				kps[0].init_idxs();		// Clear up the references to world points

				world_pts.clear();			// Clear the vector of pointers

				// Set camera poses to start again
				cam_poses[0] = Eigen::Matrix4f::Identity();
				concat_VO_pose = cam_poses[0];
				
				frame = 1;

			}
		}
	}
}

// Used to verify existence of a folder for outputting data to
void MyOdometry::VerifyOutputDir(char* outfolderDir, char* outfolderName) {

	char fullFolderPath[256];
	sprintf_s(fullFolderPath, "%s/%s", outfolderDir, outfolderName);
	boost::filesystem::path dir(fullFolderPath);
	char c;
	if (!boost::filesystem::is_directory(boost::filesystem::path(outfolderDir))) {
		std::cout << "Could not find root folder for output" << std::endl;
		std::cout << "Do you want to create root folder '" << outfolderDir
			<< "' with experiment directory " << outfolderName << "' ? [y/n]: ";

		std::cin >> c;
		if (c == 'n')
			exit(EXIT_FAILURE);
		else if (c == 'y')
			boost::filesystem::create_directories(dir);
	}
	else if (!boost::filesystem::is_directory(dir)) {
		std::cout << "Could not find experiment folder for output" << std::endl;
		std::cout << "Do you want to create experiment folder '" << outfolderName << "'? [y/n]:";
		std::cin >> c;
		if (c == 'n')
			exit(EXIT_FAILURE);
		else if (c == 'y')
			boost::filesystem::create_directory(dir);
	}
	else {
		std::cout << fullFolderPath << std::endl;
		if (boost::filesystem::is_empty(dir)) {
			std::cout << "Output folder " << outfolderName << " is currently empty. Beginning processing." << std::endl;
		}
		else {
			std::cout << "Output folder " << outfolderName << "already exists and contains data." << std::endl;
			std::cout << "Overwrite existing data ? [y / n] : ";
			std::cin >> c;
			if (c == 'n')
				exit(EXIT_FAILURE);
		}
	}

}

void MyOdometry::OpenOutputFiles(ofstream& rgbdPosesFile, ofstream& sbaPosesFile, ofstream& poseTimesFile,
	ofstream& ahrsFile, ofstream& ahrsTimes, ofstream& rawImmuOutFile, ofstream& calibImmuOutFile,
	std::ofstream& keypointCountsFiles, std::ofstream& icpfForwardPosesFile, std::ofstream& icpReversePosesFile, 
	std::ofstream& depthImStatsFile, char* outfolderDir, char* outfolderName) {
	char rgbdPosesFileName[256], sbaPosesFileName[256], poseTimesFileName[256], ahrsFileName[256],
		ahrsTimesFileName[256], rawImmuOutFileName[256], calibImmuOutFileName[256],
		keypointCountsFileName[256], icpfForwardPosesFileName[256], icpReversePosesFileName[256],
		depthImStatsFileName[256];
	sprintf_s(rgbdPosesFileName, "%s/%s/RGBDposes.txt", outfolderDir, outfolderName);
	sprintf_s(sbaPosesFileName, "%s/%s/SBAposes.txt", outfolderDir, outfolderName);
	sprintf_s(poseTimesFileName, "%s/%s/Posetimestamps.txt", outfolderDir, outfolderName);
	sprintf_s(ahrsFileName, "%s/%s/AHRSrotations.txt", outfolderDir, outfolderName);
	sprintf_s(ahrsTimesFileName, "%s/%s/AHRStimes.txt", outfolderDir, outfolderName);
	sprintf_s(rawImmuOutFileName, "%s/%s/RawIMMUData.txt", outfolderDir, outfolderName);
	sprintf_s(calibImmuOutFileName, "%s/%s/CalibIMMUData.txt", outfolderDir, outfolderName);
	sprintf_s(keypointCountsFileName, "%s/%s/KpCounts.txt", outfolderDir, outfolderName);
	sprintf_s(icpfForwardPosesFileName, "%s/%s/ICPforward.txt", outfolderDir, outfolderName);
	sprintf_s(icpReversePosesFileName, "%s/%s/ICPreverse.txt", outfolderDir, outfolderName);
	sprintf_s(depthImStatsFileName, "%s/%s/DepthImStats.txt", outfolderDir, outfolderName);

	rgbdPosesFile.open(rgbdPosesFileName);
	sbaPosesFile.open(sbaPosesFileName);
	poseTimesFile.open(poseTimesFileName);
	ahrsFile.open(ahrsFileName);
	ahrsTimes.open(ahrsTimesFileName);
	rawImmuOutFile.open(rawImmuOutFileName);
	calibImmuOutFile.open(calibImmuOutFileName);
	keypointCountsFiles.open(keypointCountsFileName);
	icpfForwardPosesFile.open(icpfForwardPosesFileName);
	icpReversePosesFile.open(icpReversePosesFileName);
	depthImStatsFile.open(depthImStatsFileName);

}


vector<cv::DMatch> MyCudaOdometry::RansacFilterFeatures(vector<cv::DMatch>& projected_matches, vector<frame_kps>& kps, vector<EigenDynamicCloud>& kp_clouds,
	int frame, MyRGBD::RansacParams& ranPams, float* camArray) {

	vector<cv::DMatch> inlier_matches;
	Timer timer;

	int num_pts = projected_matches.size();
	float* kps_new_array = new float[num_pts * 2];
	for (int i = 0; i < num_pts; i++) {
		kps_new_array[i] = kps[frame].keypoints[projected_matches[i].queryIdx].pt.y;
		kps_new_array[i + num_pts] = kps[frame].keypoints[projected_matches[i].queryIdx].pt.x;

	}
	MyCudaUtils::RansacLoadCloud(kp_clouds[frame - 1].data(), kps_new_array, num_pts, camArray, ranPams.inlier_thresh_sqd);
	delete[] kps_new_array;

	// Initialise distance array
	Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
	double* distArray = new double[num_pts];
	for (int i = 0; i < num_pts; i++) {
		distArray[i] = ranPams.inlier_thresh_sqd + 1;
	}

	EigenDynamicCloud old_sample, new_sample;
	old_sample.resize(3, 3); new_sample.resize(3, 3);

	vector<int> active_max_idxs;
	vector<int> active_curr_idxs, active_prev_idxs;
	int inliers_max = 0, inliers_curr = 0;
	for (int j = 0; j < ranPams.ransac_iterations; j++) {
		// Start the reprojection kernel for current hypothesis
		Eigen::Matrix4f invTrans = trans.inverse();
		MyCudaUtils::RansacCalcReprojDist(invTrans.data());

		// Find a random sample for the next test
		active_prev_idxs = active_curr_idxs;
		active_curr_idxs = MyRGBD::GetRandomSample(num_pts, ranPams.min_sample_size);

		//timer.tic();
		for (int i = 0; i < 3; i++) {
			old_sample.col(i) = kp_clouds[frame - 1].col(active_curr_idxs[i]);
			new_sample.col(i) = kp_clouds[frame].col(active_curr_idxs[i]);
		}
		//timer.toc();

		// Find transform hypothesis for next test
		MyRGBD::EstimateTransform(new_sample, old_sample, trans);

		// Download the current results
		MyCudaUtils::RansacGetInlierCount(&inliers_curr, distArray);

		// Get inliers for last set
		if (inliers_curr > inliers_max && j != 0) {
			active_max_idxs = active_prev_idxs;
			inliers_max = inliers_curr;
		}

	}
	// Return an empty set if no inliers found
	if (active_max_idxs.size() < 3) return inlier_matches;
	for (int i = 0; i < 3; i++) {
		old_sample.col(i) = kp_clouds[frame - 1].col(active_max_idxs[i]);
		new_sample.col(i) = kp_clouds[frame].col(active_max_idxs[i]);
	}
	MyRGBD::EstimateTransform(new_sample, old_sample, trans);
	Eigen::Matrix4f invTrans = trans.inverse();
	MyCudaUtils::RansacCalcReprojDist(invTrans.data());
	MyCudaUtils::RansacGetDists(distArray);
	MyCudaUtils::RansacTidy();


	vector<int> inlier_idxs;
	for (int i = 0; i < num_pts; i++) {
		if (distArray[i] < ranPams.inlier_thresh_sqd) {
			inlier_idxs.push_back(i);
		}
	}

	EigenDynamicCloud inlier_cloud_new, inlier_cloud_old;
	inlier_cloud_new.resize(3, inlier_idxs.size());
	inlier_cloud_old.resize(3, inlier_idxs.size());

	
	for (int i = 0; i < inlier_idxs.size(); i++)  {

		inlier_matches.push_back(projected_matches[inlier_idxs[i]]);

		inlier_cloud_new.col(i) = kp_clouds[frame].col(inlier_idxs[i]);

		inlier_cloud_old.col(i) = kp_clouds[frame - 1].col(inlier_idxs[i]);

	}


	kp_clouds[frame - 1] = inlier_cloud_old;
	kp_clouds[frame] = inlier_cloud_new;

	delete[] distArray;

	return inlier_matches;

}