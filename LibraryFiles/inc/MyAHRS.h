

#ifndef MY_AHRS_H
#define MY_AHRS_H

#define DEBUG__NO_DRIFT_CORRECTION 0	// 1 for no drift correction
										// NOTE WE REALLY WANT DRIFT CORRECTION
										// DRIFT CORRECTION MEANS USING THE ACCEL AND MAG

/***************************************************************************
*			             CALIBRATION PARAMETERS
***************************************************************************/

// Extended magnetometer calibration
#define CALIBRATION__MAGN_USE_EXTENDED true
const float magn_ellipsoid_center[3] = {96.8200, -17.1949, 23.2064};
const float magn_ellipsoid_transform[3][3] = {{0.923526, 0.00744122, 0.00225208}, {0.00744122, 0.945201, -0.00683162}, {0.00225208, -0.00683162, 0.999125}};

// Accelerometer scaling
#define ACCEL_X_MIN ((float) -266)
#define ACCEL_X_MAX ((float) 252)
#define ACCEL_Y_MIN ((float) -255)
#define ACCEL_Y_MAX ((float) 265)
#define ACCEL_Z_MIN ((float) -273)
#define ACCEL_Z_MAX ((float) 236)


// Gyroscope bias offset
// "gyro x,y,z (current/average) = .../OFFSET_X  .../OFFSET_Y  .../OFFSET_Z
#define GYRO_AVERAGE_OFFSET_X ((float) -53.0)
#define GYRO_AVERAGE_OFFSET_Y ((float) 81.0)
#define GYRO_AVERAGE_OFFSET_Z ((float) -26.0)

// Magnetometer (standard calibration mode) - REDUNDANT BECUASE OF ABOVE
// "magn x,y,z (min/max) = X_MIN/X_MAX  Y_MIN/Y_MAX  Z_MIN/Z_MAX"
#define MAGN_X_MIN ((float) -600)
#define MAGN_X_MAX ((float) 600)
#define MAGN_Y_MIN ((float) -600)
#define MAGN_Y_MAX ((float) 600)
#define MAGN_Z_MIN ((float) -600)
#define MAGN_Z_MAX ((float) -600)


/***************************************************************************
*			             SCALING PARAMETERS
***************************************************************************/

// Sensor calibration scale and offset values
#define ACCEL_X_OFFSET ((ACCEL_X_MIN + ACCEL_X_MAX) / 2.0f)
#define ACCEL_Y_OFFSET ((ACCEL_Y_MIN + ACCEL_Y_MAX) / 2.0f)
#define ACCEL_Z_OFFSET ((ACCEL_Z_MIN + ACCEL_Z_MAX) / 2.0f)
#define ACCEL_X_SCALE (GRAVITY / (ACCEL_X_MAX - ACCEL_X_OFFSET))
#define ACCEL_Y_SCALE (GRAVITY / (ACCEL_Y_MAX - ACCEL_Y_OFFSET))
#define ACCEL_Z_SCALE (GRAVITY / (ACCEL_Z_MAX - ACCEL_Z_OFFSET))

#define MAGN_X_OFFSET ((MAGN_X_MIN + MAGN_X_MAX) / 2.0f)
#define MAGN_Y_OFFSET ((MAGN_Y_MIN + MAGN_Y_MAX) / 2.0f)
#define MAGN_Z_OFFSET ((MAGN_Z_MIN + MAGN_Z_MAX) / 2.0f)
#define MAGN_X_SCALE (100.0f / (MAGN_X_MAX - MAGN_X_OFFSET))
#define MAGN_Y_SCALE (100.0f / (MAGN_Y_MAX - MAGN_Y_OFFSET))
#define MAGN_Z_SCALE (100.0f / (MAGN_Z_MAX - MAGN_Z_OFFSET))

// Gain for gyroscope (ITG-3200)
#define GYRO_GAIN 0.06957 // Same gain on all axes
#define GYRO_SCALED_RAD(x) (x * TO_RAD(GYRO_GAIN)) // Calculate the scaled gyro readings in radians per second


/***************************************************************************
*			             OTHER DCM PARAMETERS
***************************************************************************/
#define Kp_ROLLPITCH 0.02f 
#define Ki_ROLLPITCH 0.00002f
#define Kp_YAW 1.2f
#define Ki_YAW 0.00002f

// Stuff
#define GRAVITY 256.0f // "1G reference" used for DCM filter and accelerometer calibration
#define TO_RAD(x) (x * 0.01745329252)  // *pi/180
#define TO_DEG(x) (x * 57.2957795131)  // *180/pi


inline float constrain(float x, float min, float max) { return x < min ? min : (x > max ? max : x); }

#include <iostream>

class MyAHRS {
public:
	// Constructor needs input of whether raw values are being input
	MyAHRS(bool raw_inputs = true);
	~MyAHRS();
	
	void WriteRotation(std::ostream& output, float transform[3][3]);

	void WriteSensors(std::ostream& output, long long time, float accel[3], float gyro[3], float mag[3]);

	bool ScanImmuLine(std::string dataLine_in, long long& timestamp_out, float accel_out[3], float gyro_out[3], float mag_out[3] );

	// Initialisation estimates roll pitch and yaw purely of acceleration and magnetic vectors
	void init(float accel_in[3], float mag_in[3], long long time);

	// Estimation fuses gyro data with accel and mag vectors
	// Output rotation is provided as a DCM
	void estimate(float accel_in[3], float gyro_in[3], float mag_in[3], long long time, float dcm_out[3][3]);

	// Overload to the above function to output Euler angles for more readable vieweing
	// Order is Yaw-Pitch-Roll
	void estimate(float accel_in[3], float gyro_in[3], float mag_in[3], long long time, float euler_out[3]);

	// Returns the current accel, gryo and mag vectors
	// If called after an estimate function, these will be the calibrated sensor values
	void GetCalibSensors(float accel_out[3], float gyro_out[3], float mag_out[3]);

	void Compass_Heading(void);
	void Normalize(void);
	void Drift_correction(void);
	void Matrix_update(void);
	void Euler_angles(void);
	void compensate_sensor_errors(void);

private:
	// Setting for whether we should apply sensor calibration  
	bool raw_inputs;
	
	// Sensor variables
	float accel[3];  // Actually stores the NEGATED acceleration (equals gravity, if board not moving).
	float magnetom[3];
	float magnetom_tmp[3];

	float gyro[3];
	float gyro_average[3];
	int   gyro_num_samples;

	// DCM variables
	float MAG_Heading;
	float Accel_Vector[3];	// Store the acceleration in a vector
	float Gyro_Vector[3];		// Store the gyros turn rate in a vector
	float Omega_Vector[3];	// Corrected Gyro_Vector data
	float Omega_P[3];			// Omega Proportional correction
	float Omega_I[3];			// Omega Integrator
	float Omega[3];
	float errorRollPitch[3];
	float errorYaw[3];
	float DCM_Matrix[3][3];
	float Update_Matrix[3][3];
	float Temporary_Matrix[3][3];

	// Euler angles
	float yaw;
	float pitch;
	float roll;

	// DCM timing in the main loop
	long long timestamp;
	long long timestamp_old;
	float G_Dt; // Integration time for DCM algorithm

	std::string dataFormat;
};



#endif