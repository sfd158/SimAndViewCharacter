#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>

# if defined _WIN32 || defined __CYGWIN__
#   define EXPORT_API __declspec(dllexport)
# else
#   define EXPORT_API  __attribute__ ((visibility("default")))
# endif

// Note: Eigen default is col major, while pytorch is row major.
// for reading data from pytorch network, use row major here.
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> MatrixXf;
typedef Eigen::VectorXf VectorXf;
// typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3f;
typedef Eigen::Matrix3f Matrix3f;
typedef Eigen::Vector3f Vector3f;
typedef Eigen::Quaternionf Quaternionf;
typedef Eigen::Matrix4f Matrix4f;


typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXd;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Vector3d Vector3d;
typedef Eigen::Quaterniond Quaterniond;