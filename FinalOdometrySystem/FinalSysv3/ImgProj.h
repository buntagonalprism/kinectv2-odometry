
#ifndef IMG_PROJ_H
#define IMG_PROJ_H

#include "sba.h"
#include <math.h>

#define FULLQUATSZ     4

//void calcImgProj(double a[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]);
void calcImgProjFullR(double a[5], double qr0[4], double t[3], double M[3], double n[2]);

//void calcImgProjJacKRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11], double jacmS[2][3]);
//void calcImgProjJacKRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11]);
//void calcImgProjJacS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);

void calcImgProjJacRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6], double jacmS[2][3]);

//void calcImgProjJacRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6]);
//void calcDistImgProj(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]);
//void calcDistImgProjFullR(double a[5], double kc[5], double qr0[4], double t[3], double M[3], double n[2]);
//void calcDistImgProjJacKDRTS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16], double jacmS[2][3]);
//void calcDistImgProjJacKDRT(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16]);
//void calcDistImgProjJacS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);

void img_projsRTS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata);

void img_projsRTS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata);

#define _MK_QUAT_FRM_VEC(q, v){                                     \
	(q)[1] = (v)[0]; (q)[2] = (v)[1]; (q)[3] = (v)[2];                      \
	(q)[0] = sqrt(1.0 - (q)[1] * (q)[1] - (q)[2] * (q)[2] - (q)[3] * (q)[3]);  \
}

inline void quatMultFast(double q1[FULLQUATSZ], double q2[FULLQUATSZ], double p[FULLQUATSZ])
{
	double t1, t2, t3, t4, t5, t6, t7, t8, t9;
	//double t10, t11, t12;

	t1 = (q1[0] + q1[1])*(q2[0] + q2[1]);
	t2 = (q1[3] - q1[2])*(q2[2] - q2[3]);
	t3 = (q1[1] - q1[0])*(q2[2] + q2[3]);
	t4 = (q1[2] + q1[3])*(q2[1] - q2[0]);
	t5 = (q1[1] + q1[3])*(q2[1] + q2[2]);
	t6 = (q1[1] - q1[3])*(q2[1] - q2[2]);
	t7 = (q1[0] + q1[2])*(q2[0] - q2[3]);
	t8 = (q1[0] - q1[2])*(q2[0] + q2[3]);

	/* following fragment it equivalent to the one above */
	t9 = 0.5*(t5 - t6 + t7 + t8);
	p[0] = t2 + t9 - t5;
	p[1] = t1 - t9 - t6;
	p[2] = -t3 + t9 - t8;
	p[3] = -t4 + t9 - t7;
}

struct globs_{
	double *rot0params; /* initial rotation parameters, combined with a local rotation parameterization */
	double intrcalib[5]; /* the 5 intrinsic calibration parameters in the order [fu, u0, v0, ar, skew],
					   * where ar is the aspect ratio fv/fu. Use when calib params are fixed */

	int nccalib; /* number of calibration parameters that must be kept constant.
				 * Used only when calibration varies among cameras */

	int ncdist; /* number of distortion parameters in Bouguet's model that must be kept constant.
				* Used only when calibration varies among cameras and distortion is to be estimated */
	int cnp, pnp, mnp; /* dimensions */

	double *ptparams; /* needed only when bundle adjusting for camera parameters only */
	double *camparams; /* needed only when bundle adjusting for structure parameters only */
};

#endif

