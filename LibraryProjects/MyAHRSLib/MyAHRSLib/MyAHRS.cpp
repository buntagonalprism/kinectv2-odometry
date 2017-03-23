
#include "MyAHRS.h"
#include "DCM.h"
#include "Compass.h"


void MyAHRS::WriteRotation(std::ostream& output, float transform[3][3]) {
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) 
			output << transform[x][y] << " ";
		output << std::endl;
	}
	output << std::endl;
}

void MyAHRS::WriteSensors(std::ostream& output, long long time, float accel[3], float gyro[3], float mag[3] ) {
	output << time << " ";
	output << accel[0] << " " << accel[1] << " " << accel[2] << " "
		<< gyro[0] << " " << gyro[1] << " " << gyro[2] << " "
		<< mag[0] << " " << mag[1] << " " << mag[2] << std::endl;
}

float norm3(float x1, float x2, float x3) {
	return sqrt(x1*x1 + x2*x2 + x3*x3);
}

//void 
bool MyAHRS::ScanImmuLine(std::string dataLine, long long& timestamp_out, float accel_out[3], float gyro_out[3], float mag_out[3] ) {
	float ax, ay, az, mx, my, mz, gx, gy, gz;
	if (sscanf(dataLine.c_str(), dataFormat.c_str(),
		&timestamp_out, &ax, &ay, &az, &mx, &my, &mz, &gx, &gy, &gz) == 10) {

		// Sanity check on received data
		// Reject large spikes in all sensors
		// Reject norm of accel and mag dropping too small
		if (abs(ax) < 1000 && abs(ay) < 1000 && abs(az) < 1000 && abs(gx) < 10000 && abs(gy) < 10000 && abs(gz) < 10000 && abs(mx) < 2000 && abs(my) < 2000 && abs(mz) < 2000
			&& norm3(ax,ay,az) > 100 && norm3(mx,my,mz) > 100) {
			accel_out[0] = ax;  accel_out[1] = ay;  accel_out[2] = az;
			mag_out[0] = mx;    mag_out[1] = my;    mag_out[2] = mz;
			gyro_out[0] = gx;   gyro_out[1] = gy;   gyro_out[2] = gz;

			return true;
		}

	}
	
	return false;
}

void MyAHRS::estimate(float accel_in[3], float gyro_in[3], float mag_in[3], long long time, float dcm_out[3][3]) {
	// Copy data into local memory
	accel[0] = accel_in[0];
	accel[1] = accel_in[1];
	accel[2] = accel_in[2];
	gyro[0] = gyro_in[0];
	gyro[1] = gyro_in[1];
	gyro[2] = gyro_in[2];
	magnetom[0] = mag_in[0];
	magnetom[1] = mag_in[1];
	magnetom[2] = mag_in[2];

	// Find DeltaT
	timestamp = time;
	G_Dt = (timestamp - timestamp_old)/1000000.0;
	timestamp_old = timestamp;

	if (raw_inputs)
		compensate_sensor_errors();

	Compass_Heading();
	Matrix_update();
	Normalize();
	Drift_correction();
	Euler_angles();
	
	for (int x = 0; x<3; x++) // Output DCM
		for (int y = 0; y<3; y++)
			dcm_out[x][y] = DCM_Matrix[x][y];
		
}

void MyAHRS::GetCalibSensors(float accel_out[3], float gyro_out[3], float mag_out[3]) {
	accel_out[0] = accel[0];
	accel_out[1] = accel[1];
	accel_out[2] = accel[2];
	gyro_out[0] = gyro[0];
	gyro_out[1] = gyro[1];
	gyro_out[2] = gyro[2];
	mag_out[0] = magnetom[0];
	mag_out[1] = magnetom[1];
	mag_out[2] = magnetom[2];
}

void MyAHRS::estimate(float accel_in[3], float gyro_in[3], float mag_in[3], long long time, float euler_out[3]) {
	float dcm[3][3];
	estimate( accel_in,  gyro_in,  mag_in,   time, dcm);
	euler_out[0] = TO_DEG(yaw);
	euler_out[1] = TO_DEG(pitch);
	euler_out[2] = TO_DEG(roll);
}

MyAHRS::MyAHRS(bool raw_inputs_in ) {
	for (int i = 0; i < 3; i++) {
		Accel_Vector[i] = 0;
		Gyro_Vector[i] = 0;
		Omega_Vector[i] = 0;
		Omega_P[i] = 0;
		Omega_I[i] = 0;
		Omega[i] = 0;
		errorRollPitch[i] =0;
		errorYaw[i] = 0;
	}
	EyeMatrix(DCM_Matrix);
	ZeroMatrix(Temporary_Matrix);

	raw_inputs = raw_inputs_in;

	if (raw_inputs)
		dataFormat = "i %I64d #A-R=%f,%f,%f #M-R=%f,%f,%f #G-R=%f,%f,%f";
	else
		dataFormat = "i %I64d #A-C=%f,%f,%f #M-C=%f,%f,%f #G-C=%f,%f,%f";


	int k = 0;
	for (int i = 0; i < 3; i ++)
		for (int j = 0; j < 3; j++)
			Update_Matrix[i][j] = k++;
	
}

void MyAHRS::init(float accel_in[3], float mag_in[3], long long time ){
	accel[0] = accel_in[0];
	accel[1] = accel_in[1];
	accel[2] = accel_in[2];
	magnetom[0] = mag_in[0];
	magnetom[1] = mag_in[1];
	magnetom[2] = mag_in[2];

	timestamp_old = time;

	float temp1[3];
	float temp2[3];
	float xAxis[] = { 1.0f, 0.0f, 0.0f };

	// GET PITCH
	// Using y-z-plane-component/x-component of gravity vector
	pitch = -atan2(accel[0], sqrt(accel[1] * accel[1] + accel[2] * accel[2]));

	// GET ROLL
	// Compensate pitch of gravity vector 
	Vector_Cross_Product(temp1, accel, xAxis);
	Vector_Cross_Product(temp2, xAxis, temp1);
	// Normally using x-z-plane-component/y-component of compensated gravity vector
	// roll = atan2(temp2[1], sqrt(temp2[0] * temp2[0] + temp2[2] * temp2[2]));
	// Since we compensated for pitch, x-z-plane-component equals z-component:
	roll = atan2(temp2[1], temp2[2]);

	// GET YAW
	Compass_Heading();
	yaw = MAG_Heading;

	// Init rotation matrix
	init_rotation_matrix(DCM_Matrix, yaw, pitch, roll);
}

MyAHRS::~MyAHRS() {}


// Apply calibration to raw sensor readings
void MyAHRS::compensate_sensor_errors(void) {
	// Compensate accelerometer error
	accel[0] = (accel[0] - ACCEL_X_OFFSET) * ACCEL_X_SCALE;
	accel[1] = (accel[1] - ACCEL_Y_OFFSET) * ACCEL_Y_SCALE;
	accel[2] = (accel[2] - ACCEL_Z_OFFSET) * ACCEL_Z_SCALE;

	// Compensate magnetometer error
#ifdef CALIBRATION__MAGN_USE_EXTENDED 
	for (int i = 0; i < 3; i++)
		magnetom_tmp[i] = magnetom[i] - magn_ellipsoid_center[i];
	Matrix_Vector_Multiply(magn_ellipsoid_transform, magnetom_tmp, magnetom);
#else
	magnetom[0] = (magnetom[0] - MAGN_X_OFFSET) * MAGN_X_SCALE;
	magnetom[1] = (magnetom[1] - MAGN_Y_OFFSET) * MAGN_Y_SCALE;
	magnetom[2] = (magnetom[2] - MAGN_Z_OFFSET) * MAGN_Z_SCALE;
#endif

	// Compensate gyroscope error
	gyro[0] -= GYRO_AVERAGE_OFFSET_X;
	gyro[1] -= GYRO_AVERAGE_OFFSET_Y;
	gyro[2] -= GYRO_AVERAGE_OFFSET_Z;
}