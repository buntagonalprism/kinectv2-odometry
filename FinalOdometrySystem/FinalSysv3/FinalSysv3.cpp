
#include "MyOdometry.h"


int main(void) {
	// Use these ones for the nice rocks I've been using
	//char folderDir[] = "G:/adatasets/14_10_2_ColourHarold";
	//char folderName[] = "06_lastrocks";

	char folderDir[] = "G:/adatasets/undated";
	char folderName[] = "shortypresent";

	char outfolderDir[] = "D:/Dropbox/Uni Work/Thesis - Alex Bunting/03-Software Design/07-VoOutputs/14_9_28_TunksImmuCalib";
	char outfolderName[] = "04_justforimages";

	{
		MyOdometry odo(10);

		odo.InitDataSource(folderDir, folderName, OBS_COLOUR_AND_DEPTH);

		odo.InitOutput(outfolderDir, outfolderName);

		odo.InitStarTracker(128, 30, 10, 8, 10);

		odo.InitBriefDescriptor();

		odo.InitRansac();
		
		odo.InitSBA();

		//odo.Start(22);

		odo.StartExperimental();

	}
}




