

// Structure stores information about all unique world points
struct world_pt {
public:
	float x, y, z;		// stores 3D world coordinates
	int num_vis;		// Sum of how many frames it is visible in
	int* frames_vis;	// Array for mask for which frames its visible in
	int* row_meas;		// row coordinates for each image observation
	int* col_meas;		// column coordinates for each image observation

	// Constructor allocates memory and initialises arrays
	world_pt(int numframes = NUM_FRAMES) : x(0.0f), y(0.0f), z(0.0f), num_vis(0),
		frames_vis(NULL), row_meas(NULL), col_meas(NULL) {
		frames_vis = new int[numframes];
		row_meas = new int[numframes];
		col_meas = new int[numframes];

		for (int i = 0; i < numframes; i++) {
			frames_vis[i] = 0;
			row_meas[i] = -1;
			col_meas[i] = -1;
		}
	}

	// Destructor frees memory
	~world_pt() {
		if (frames_vis)
			delete[] frames_vis;
		if (row_meas)
			delete[] row_meas;
		if (col_meas)
			delete[] col_meas;
	}
};

// Structure stores information about keypoints in a frame
struct frame_kps {
public:
	vector<cv::KeyPoint> keypoints;
	int* world_idxs;

	void init_idxs(void) {
		world_idxs = new int[keypoints.size()];
		for (int i = 0; i < keypoints.size(); i++) {
			world_idxs[i] = -1;
		}
	}
	~frame_kps() {
		if (world_idxs)
			delete[] world_idxs;
	}
};