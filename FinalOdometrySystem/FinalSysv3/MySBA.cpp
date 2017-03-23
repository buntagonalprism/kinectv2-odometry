
#include "MySBA.h"

void MySBA::AddPointsAndMeasurements(
	vector<world_pt>& world_pts, 
	vector<frame_kps>& keypoints, 
	vector<cv::DMatch>& matches, 
	vector<Eigen::Matrix<float, 3, Eigen::Dynamic>>& kp_clouds, 
	vector<Eigen::Matrix4f>& cam_poses,
	int& num_meas, 
	int frame, 
	int numframes) 
{

	// Iterate through the good projected matches
	for (int i = 0; i < matches.size(); i++) {
		int m = matches[i].trainIdx;	 // previous frame keypoint index
		int n = matches[i].queryIdx;	 // current frame keypoint

		int col_old = keypoints[frame - 1].keypoints[m].pt.x;		// 
		int row_old = keypoints[frame - 1].keypoints[m].pt.y;		// 
		int col_new = keypoints[frame].keypoints[n].pt.x;			//
		int row_new = keypoints[frame].keypoints[n].pt.y;			//

		int idx;
		if ((idx = keypoints[frame - 1].world_idxs[m]) != -1) {
			// We already have a 3D point associated with the keypoint in previous frame
			// Copy through a reference to it in the keypoint vector
			keypoints[frame].world_idxs[n] = idx;
			world_pts[idx].frames_vis[frame] = 1;
			world_pts[idx].col_meas[frame] = col_new;
			world_pts[idx].row_meas[frame] = row_new;
			num_meas++;
		}
		else {
			// New 3D point
			world_pt pt;

			// TODO: check cloud transpose
			// Homogenous coordinates for local 3D keypoint location
			Eigen::Matrix<float, 4, 1> local_pt;
			local_pt(0) = (kp_clouds[frame - 1])(0, i);			// From inside to out: Get the pointer to the dynamic matrix, deref it, then access the element
			local_pt(1) = (kp_clouds[frame - 1])(1, i);
			local_pt(2) = (kp_clouds[frame - 1])(2, i);
			local_pt(3) = 1;

			// Rotate to global frame
			Eigen::Matrix<float, 4, 1> world_coord = cam_poses[frame - 1] * local_pt;
			pt.x = world_coord(0);
			pt.y = world_coord(1);
			pt.z = world_coord(2);

			// Set frames visible in the mask
			pt.frames_vis[frame - 1] = 1;
			pt.frames_vis[frame] = 1;

			// Assign the measurements
			pt.row_meas[frame] = row_new;
			pt.col_meas[frame] = col_new;
			pt.row_meas[frame - 1] = row_old;
			pt.col_meas[frame - 1] = col_old;

			// In the keypoint vector, set which 3D point index they are associated with 
			// (i.e. the one at the end of the current list)
			keypoints[frame].world_idxs[n] = world_pts.size();
			keypoints[frame - 1].world_idxs[m] = world_pts.size();

			// Add it to the set of 3D points
			world_pts.push_back(std::move(pt));

			// Increment number of measurements
			num_meas += 2;

		}
	}

}

void MySBA::BundleAdjust(vector<world_pt>& world_pts, vector<Eigen::Matrix4f>& cam_poses, vector<Eigen::Matrix4f>& ba_poses, int num_meas, SBAParams& sbaParams) {

	// Local vars for readability
	int numframes = sbaParams.numframes;
	int cnp = sbaParams.cnp, pnp = sbaParams.pnp, mnp = sbaParams.mnp;
	globs_ globs = sbaParams.globs;
	int verbose = sbaParams.verbose;
	double* opts = sbaParams.opts;
	double* info = sbaParams.info;


	// Allocate optimisation variable
	double* sba_opt_var = new double[numframes*cnp + world_pts.size()*pnp];

	// Allocate mask for every 3D point whether it is visible in every frame
	char* vmask = new char[numframes * world_pts.size()];

	// Allocate measurements (row,col) for every keypoint observation of a 3D point
	double* measurements = new double[mnp * num_meas];

	// Allocate initial quaternion rotations
	double* initrot = new double[4 * numframes];

	// Pointers to access arrays element-wise
	double* pOptVar = sba_opt_var;
	char* pvmask = vmask;
	double* pmeas = measurements;

	// Concatenate all the estimated camera poses in required format
	for (int i = 0; i < numframes; i++) {
		// Get thepose
		Eigen::Affine3f tmp = Eigen::Affine3f(cam_poses[i]);
		Eigen::Affine3f pose = tmp.inverse();
		float qvec[3] = { 0.0 }, trans[3] = { 0.0 };

		// Convert rotation matrix to qVec
		Rot2QVec(pose.rotation(), qvec);

		// Store rotation and translation in sba_opt_var
		*pOptVar++ = qvec[0];
		*pOptVar++ = qvec[1];
		*pOptVar++ = qvec[2];
		*pOptVar++ = pose(0, 3);
		*pOptVar++ = pose(1, 3);
		*pOptVar++ = pose(2, 3);
	}

	// Iterate through all world_pts
	for (int i = 0; i < world_pts.size(); i++) {
		// Concatenate all the 3D pts into sba_opt_var
		*pOptVar++ = world_pts[i].x;
		*pOptVar++ = world_pts[i].y;
		*pOptVar++ = world_pts[i].z;

		// Concatenate all the masks into vmask (visibility mask)
		for (int j = 0; j < numframes; j++) {
			*pvmask++ = world_pts[i].frames_vis[j];
		}

		// Concatenate all the valid measurements (where mask = 1) in order
		for (int j = 0; j < numframes; j++) {
			if (world_pts[i].frames_vis[j] == 1) {
				*pmeas++ = world_pts[i].col_meas[j];
				*pmeas++ = world_pts[i].row_meas[j];
			}
		}
	}

	// Create the covarience matrix - i.e. expected noise on each measurement. Don't have data on this though
	double* covimgpts = NULL;

	// Copy the rotations to a seperate array, and clear the ones in the optimisation vector
	// Don't know why. Also the original notation used in SBA is a mess to read
	for (int i = 0; i < numframes; ++i){
		int j = (i + 1)*cnp; // note the +1, below we move from right to left, assuming 3 parameters for the translation!
		initrot[(4 * i) + 1] = sba_opt_var[j - 6];
		initrot[(4 * i) + 2] = sba_opt_var[j - 5];
		initrot[(4 * i) + 3] = sba_opt_var[j - 4];
		initrot[4 * i] = sqrt(1.0 - sba_opt_var[j - 6] * sba_opt_var[j - 6] - sba_opt_var[j - 5] * sba_opt_var[j - 5] - sba_opt_var[j - 4] * sba_opt_var[j - 4]);
		sba_opt_var[j - 4] = sba_opt_var[j - 5] = sba_opt_var[j - 6] = 0.0; // clear rotation
	}
	globs.rot0params = initrot;

	// Call sba finally using the abridged version in the comments
	int n = sba_motstr_levmar_x(world_pts.size(), 0, numframes, 1, vmask, sba_opt_var, cnp, pnp, measurements, covimgpts, mnp,
		img_projsRTS_x, img_projsRTS_jac_x, (void *)(&globs), MAXITER2, verbose, opts, info);

	// Get the output poses and transform back to rotationMats
	pOptVar = sba_opt_var;
	for (int i = 0; i < numframes; i++) {
		Eigen::Matrix3d rotation;
		Eigen::Matrix4f transf;
		double qvec[3] = { 0.0 }, transl[3] = { 0.0 };
		qvec[0] = *pOptVar++;
		qvec[1] = *pOptVar++;
		qvec[2] = *pOptVar++;


		// Don't quite understand the purpose of combining these rotations -
		// Apparently its compounding local rotation estimtes with initial ones
		double qs[FULLQUATSZ], *q0, prd[FULLQUATSZ];

		_MK_QUAT_FRM_VEC(qs, qvec);

		q0 = initrot + i*FULLQUATSZ;
		quatMultFast(qs, q0, prd); // prd=qs*q0

		/* copy back vector part making sure that the scalar part is non-negative */
		if (prd[0] >= 0.0){
			qvec[0] = prd[1];  qvec[1] = prd[2];  qvec[2] = prd[3];
		}
		else{ // negate since two quaternions q and -q represent the same rotation
			qvec[0] = -prd[1];  qvec[1] = -prd[2];  qvec[2] = -prd[3];
		}

		transl[0] = *pOptVar++;
		transl[1] = *pOptVar++;
		transl[2] = *pOptVar++;
		QVecTrans2Homog(qvec, transl, transf);
		//std::cout << transf << std::endl;
		//if (i > 0) {
		//
		//	relative_sba_pose = ba_poses[i - 1].inverse()*transf;
		//	sbaPosesFile << relative_sba_pose << std::endl;
		//}
		ba_poses[i] = transf;

	}

	// Tidy up the allocated data
	delete[] sba_opt_var;
	delete[] measurements;
	delete[] vmask;
	delete[] initrot;

}

void MySBA::WriteRelativePoses(std::ostream& output, vector<Eigen::Matrix4f> ba_poses) {
	
	for (int i = 1; i < ba_poses.size(); i++) {
	
		Eigen::Matrix4f relative_sba_pose = ba_poses[i - 1].inverse()*ba_poses[i];
		output << relative_sba_pose << std::endl;

	}
}