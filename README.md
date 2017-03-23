# kinectv2-odometry

Long range position tracking (odometry) using the Kinect for Windows V2 colour and depth (RGBD) camera and an inertial and magnetic measurement unit (IMMU). As described in more detail in the documentation, odometry is first performed by detecting and matching keypoints across successive colour image frames and estimating camera pose change by projecting those keypoints to 3D using the depth image and aligning the point clouds. This is then used as an initial estimate for an alignment using the full point cloud projected from the depth image using iterative closest point. The IMMU data is combined to make an attitude heading reference system which is fused with the camera pose estimates to provide some global orientation accuracy.  

## Documentation
Please see the 'Documentation' folder for the following documentation:
- High level poster summarising the experiment goal and results
- Journal-format paper with further details on equations and methods
- Full thesis document with all technical method and results included 

## Internal Dependencies
The final project has a number of dependent projects contained in 'LibraryProjects' folder:
- AHRS: Takes input received from an IMMU and calculates attitude and heading
- Arduino: project controlls serial interface with an arduino microcontroller, intending to be collecting IMMU data
- CudaUtils: various CUDA-optimised algorithms for GPU including BRIEF descriptor and STAR keypoint computation
- KinectUtils: utility functions for mapping between colour and depth image spaces and 3D coordinates
- Kinectv2: wrapper around Kinect for Windows SDK including utilities for starting and stopping image capture streams and capturing single images with both rgb and depth cameras
- RGBD: Perform RGBD odometry by first applying visual odometry with keypoint tracking then refining using iterative closest point
- Simulator: simulates a kinect rgb-depth image stream by reading of pre-captured files from the filesystem

## External Dependencies
A number of external library dependencies are required to build and run kinect RGBD odometry
- Point cloud library: for 3D point cloud manipulations
- Boost: various helper libraries for C++
- Eigen: matrix mathematics library
- Kinect SDK: Drivers for Kinect for Windows Camera
- SBA: Sparse bundle adjustment library for estimating pose changes from successive images
