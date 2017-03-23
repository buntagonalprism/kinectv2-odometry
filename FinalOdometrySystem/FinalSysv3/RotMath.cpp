
#include "RotMath.h"

void Rot2Q(Eigen::Matrix<float, 3, 3> rot_in, float q_out[4]) {
	float m00 = rot_in(0, 0), m01 = rot_in(0, 1), m02 = rot_in(0, 2);
	float m10 = rot_in(1, 0), m11 = rot_in(1, 1), m12 = rot_in(1, 2);
	float m20 = rot_in(2, 0), m21 = rot_in(2, 1), m22 = rot_in(2, 2);

	float tr = m00 + m11 + m22;

	if (tr > 0) {
		float S = sqrt(tr + 1.0) * 2; // S=4*qw 
		q_out[0] = 0.25 * S;
		q_out[1] = (m21 - m12) / S;
		q_out[2] = (m02 - m20) / S;
		q_out[3] = (m10 - m01) / S;
	}
	else if ((m00 > m11)&(m00 > m22)) {
		float S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx 
		q_out[0] = (m21 - m12) / S;
		q_out[1] = 0.25 * S;
		q_out[2] = (m01 + m10) / S;
		q_out[3] = (m02 + m20) / S;
	}
	else if (m11 > m22) {
		float S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
		q_out[0] = (m02 - m20) / S;
		q_out[1] = (m01 + m10) / S;
		q_out[2] = 0.25 * S;
		q_out[3] = (m12 + m21) / S;
	}
	else {
		float S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
		q_out[0] = (m10 - m01) / S;
		q_out[1] = (m02 + m20) / S;
		q_out[2] = (m12 + m21) / S;
		q_out[3] = 0.25 * S;
	}

	NormaliseQ(q_out);
}

void Rot2QVec(Eigen::Matrix<float, 3, 3> rot_in, float qvec_out[3]) {

	float q_temp[4] = { 0.0 };

	Rot2Q(rot_in, q_temp);

	Q2QVec(q_temp, qvec_out);
}

void QVecTrans2Homog(double qvec_in[3], double trans[3], Eigen::Matrix4f& transform_out) {

	transform_out = Eigen::Matrix4f::Identity();

	double qx = qvec_in[0];
	double qy = qvec_in[1];
	double qz = qvec_in[2];

	double sqx = qx*qx;
	double sqy = qy*qy;
	double sqz = qz*qz;

	double qw = sqrt(1 - sqx - sqy - sqz);
	double sqw = qw*qw;

	// Rotation 
	// invs (inverse square length) is only required if quaternion is not already normalised
	double invs = 1 / (sqx + sqy + sqz + sqw);
	transform_out(0, 0) = (float)(sqx - sqy - sqz + sqw)*invs; // since sqw + sqx + sqy + sqz =1/invs*invs
	transform_out(1, 1) = (float)(-sqx + sqy - sqz + sqw)*invs;
	transform_out(2, 2) = (float)(-sqx - sqy + sqz + sqw)*invs;

	double tmp1 = qx*qy;
	double tmp2 = qz*qw;
	transform_out(1, 0) = (float) 2.0 * (tmp1 + tmp2)*invs;
	transform_out(0, 1) = (float) 2.0 * (tmp1 - tmp2)*invs;

	tmp1 = qx*qz;
	tmp2 = qy*qw;
	transform_out(2, 0) = (float) 2.0 * (tmp1 - tmp2)*invs;
	transform_out(0, 2) = (float) 2.0 * (tmp1 + tmp2)*invs;
	tmp1 = qy*qz;
	tmp2 = qx*qw;
	transform_out(2, 1) = (float) 2.0 * (tmp1 + tmp2)*invs;
	transform_out(1, 2) = (float) 2.0 * (tmp1 - tmp2)*invs;

	transform_out(0, 3) = (float)trans[0];
	transform_out(1, 3) = (float)trans[1];
	transform_out(2, 3) = (float)trans[2];
}

void NormaliseQ(float q_in[4]) {
	float qnorm = sqrt((q_in[0] * q_in[0]) + (q_in[1] * q_in[1]) + (q_in[2] * q_in[2]) + (q_in[3] * q_in[3]));
	q_in[0] /= qnorm;
	q_in[1] /= qnorm;
	q_in[2] /= qnorm;
	q_in[3] /= qnorm;
}

// Returns the vector part of the equivalent quaternion with a positive scalar component (since q = -q)
void Q2QVec(float q_in[4], float qvec_out[3]) {
	float factor = q_in[0] >= 0.0 ? 1.0 : -1.0;
	qvec_out[0] = q_in[1] * factor;
	qvec_out[1] = q_in[2] * factor;
	qvec_out[2] = q_in[3] * factor;
}