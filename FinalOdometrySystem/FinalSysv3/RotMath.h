#ifndef ROT_MATH_H
#define ROT_MATH_H

#include <pcl/common/eigen.h>
#include <pcl/common/impl/eigen.hpp>

// Convert rotation matrix to quaternion
void Rot2Q(Eigen::Matrix<float, 3, 3> rot_in, float q_out[4]);

// Convert a rotation matrix to a quaternion and an equiv
void Rot2QVec(Eigen::Matrix<float, 3, 3> rot_in, float qvec_out[3]);

// Normalise a quaternion
void NormaliseQ(float q_in[4]);

// Converts a quaternion to just its vector component equivalent to 
// having a positive scalar component. 
void Q2QVec(float q_in[4], float qvec_out[3]);

// Convert quaternion vector part to homogoenous transform matrix
void QVecTrans2Homog(double qvec_in[3], double trans[3], Eigen::Matrix4f& transform_out);

#endif