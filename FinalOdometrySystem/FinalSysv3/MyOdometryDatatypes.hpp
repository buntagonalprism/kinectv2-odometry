

#ifndef MY_ODOMETRY_DATATYPES_H
#define MY_ODOMETRY_DATATYPES_H

#include <vector>
#include <opencv2\features2d\features2d.hpp>


#define NUM_FRAMES 10

typedef enum 
{
	DRAW_KEYPOINTS		= 1,
	DRAW_MATHCES		= 2,
	DRAW_TRACES			= 4,
	DRAW_KP_CLOUDS		= 8,
	DRAW_DENSE_CLOUDS	= 16,
} DrawResults;

using std::vector;


// Structure stores information about all unique world points
struct world_pt {
public:
	float x, y, z;		// stores 3D world coordinates
	int num_vis;		// Sum of how many frames it is visible in
	int* frames_vis;	// Array for mask for which frames its visible in
	int* row_meas;		// row coordinates for each image observation
	int* col_meas;		// column coordinates for each image observation
	int num_frames;

	// Copy constructor
	world_pt(world_pt& other) : x(other.x), y(other.y), z(other.z), num_vis(other.num_vis), num_frames(other.num_frames) {

		frames_vis = new int[num_frames];
		row_meas = new int[num_frames];
		col_meas = new int[num_frames];

		for (int i = 0; i < num_frames; i++) {
			frames_vis[i] = other.frames_vis[i];
			row_meas[i] = other.row_meas[i];
			col_meas[i] = other.col_meas[i];
		}
	}

	// Constructor allocates memory and initialises arrays
	world_pt(int numframes = NUM_FRAMES) : num_frames (numframes), x(0.0f), y(0.0f), z(0.0f), num_vis(0),
		frames_vis(NULL), row_meas(NULL), col_meas(NULL) {
		frames_vis = new int[num_frames];
		row_meas = new int[num_frames];
		col_meas = new int[num_frames];

		for (int i = 0; i < num_frames; i++) {
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

	// Empty default constructor
	frame_kps() : world_idxs(NULL) {}

	// Copy constructor performs deep copy
	frame_kps(frame_kps& other) {
		keypoints = other.keypoints;
		world_idxs = new int[keypoints.size()];
		for (int i = 0; i < keypoints.size(); i++) {
			world_idxs[i] = other.world_idxs[i];
		}
	}

	void init_idxs(void) {
		world_idxs = new int[keypoints.size()];
		for (int i = 0; i < keypoints.size(); i++) {
			world_idxs[i] = -1;
		}
	}
	~frame_kps() {
		keypoints.clear();
		if (world_idxs)
			delete[] world_idxs;
	}
};


struct OdoOpts {
public:
	OdoOpts() : 
		verboseLvl1(false),
		verboseLvl2(false),
		verboseLvl3(false),
		timingLvl1(false),
		timingLvl2(false),
		timingLvl3(false),
		printLvl1(false),
		printLvl2(false),
		printLvl3(false),

		timeDetector(false),
		timeExtractor(false),
		timeFeatures(false),
		timeBruteForce(false),
		timeLowe(false),
		timeProjection(false),
		timeRansac(false),
		timeFiltering(false),
		timeAlignment(false),
		timeFrame(false),
		timeSbaFormat(false),
		timeSbaAdjustment(false),

		printKps(false),
		printLowe(false),
		printValidDepth(false),
		printRansac(false),
		printIncremental(false),
		printConcat(false),
		printSBA(false),
		printAhrs(false),
		printImmuRaw(false),
		printImmuCalib(false),

		drawKps(false),
		drawMatches(false),
		drawTraces(false),
		drawKpClouds(false),
		drawDenseClouds(false)
		{}
	
	bool verboseLvl1;
	bool verboseLvl2;
	bool verboseLvl3;
	bool timingLvl1;
	bool timingLvl2;
	bool timingLvl3;
	bool printLvl1;
	bool printLvl2;
	bool printLvl3;

	bool timeDetector;
	bool timeExtractor;
	bool timeFeatures;
	bool timeBruteForce;
	bool timeLowe;
	bool timeProjection;
	bool timeRansac;
	bool timeFiltering;
	bool timeAlignment;
	bool timeFrame;
	bool timeSbaFormat;
	bool timeSbaAdjustment;

	bool printKps;
	bool printLowe;
	bool printValidDepth;
	bool printRansac;
	bool printIncremental;
	bool printConcat;
	bool printSBA;
	bool printAhrs;
	bool printImmuRaw;
	bool printImmuCalib;

	bool drawKps;
	bool drawMatches;
	bool drawTraces;
	bool drawKpClouds;
	bool drawDenseClouds;

};


#endif
