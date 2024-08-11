#include <iostream>
#include <random>
#include <EigenExtension/Mat3Euler.h>
#include "Wm5/Wm5Matrix3.h"

void check(const Eigen::Matrix3d & eigen_mat3, const Wm5::Matrix3d & wm5_mat3)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			// std::cout << "(i, j) = " << "(" << i << ", " << j << ")" << "eigen " << eigen_mat3(i, j) << " Wm5 " << wm5_mat3[i][j] << std::endl;
			assert(std::abs(eigen_mat3(i, j) - wm5_mat3[i][j]) < 1e-12);
		}
	}
	std::cout << "Check OK" << std::endl;
}
int main0()
{
	// There is no bug in EulerMaker and EulerExtractor...
	std::uniform_real_distribution<double> dist{ -M_PI, M_PI };
	std::random_device rd; // Non-deterministic seed source
	std::default_random_engine rng{ rd() };
	for (int i = 0; i < 1000; i++)
	{
		double a[3] = { dist(rng), dist(rng), dist(rng) };
		double b[3] = { 0, 0, 0 };
		double c[3] = { 0, 0, 0 };
		Eigen::Matrix3d eigen_xyz = EulerMakerXYZ(a[0], a[1], a[2]);
		Wm5::Matrix3d wm5_xyz;
		wm5_xyz.MakeEulerXYZ(a[0], a[1], a[2]);
		std::cout << "Check XYZ " << a[0] << " " << a[1] << " " << a[2] << std::endl;
		check(eigen_xyz, wm5_xyz);
		wm5_xyz.ExtractEulerXYZ(b[0], b[1], b[2]);
		EulerFactorXYZ(eigen_xyz, c[0], c[1], c[2]);
		assert(std::abs(b[0] - c[0]) < 1e-12 && std::abs(b[1] - c[1]) < 1e-12 && std::abs(b[2] - c[2]) < 1e-12);

		Eigen::Matrix3d eigen_xzy = EulerMakerXZY(a[0], a[1], a[2]);
		Wm5::Matrix3d wm5_xzy;
		wm5_xzy.MakeEulerXZY(a[0], a[1], a[2]);
		std::cout << "Check XZY" << a[0] << " " << a[1] << " " << a[2] << std::endl;
		check(eigen_xzy, wm5_xzy);
		wm5_xzy.ExtractEulerXZY(b[0], b[1], b[2]);
		EulerFactorXZY(eigen_xzy, c[0], c[1], c[2]);
		assert(std::abs(b[0] - c[0]) < 1e-12 && std::abs(b[1] - c[1]) < 1e-12 && std::abs(b[2] - c[2]) < 1e-12);

		Eigen::Matrix3d eigen_yxz = EulerMakerYXZ(a[0], a[1], a[2]);
		Wm5::Matrix3d wm5_yxz;
		wm5_yxz.MakeEulerYXZ(a[0], a[1], a[2]);
		std::cout << "Check YXZ" << a[0] << " " << a[1] << " " << a[2] << std::endl;
		check(eigen_yxz, wm5_yxz);
		wm5_yxz.ExtractEulerYXZ(b[0], b[1], b[2]);
		EulerFactorYXZ(eigen_yxz, c[0], c[1], c[2]);
		assert(std::abs(b[0] - c[0]) < 1e-12 && std::abs(b[1] - c[1]) < 1e-12 && std::abs(b[2] - c[2]) < 1e-12);

		Eigen::Matrix3d eigen_yzx = EulerMakerYZX(a[0], a[1], a[2]);
		Wm5::Matrix3d wm5_yzx;
		wm5_yzx.MakeEulerYZX(a[0], a[1], a[2]);
		std::cout << "Check YZX" << a[0] << " " << a[1] << " " << a[2] << std::endl;
		check(eigen_yzx, wm5_yzx);
		wm5_yzx.ExtractEulerYZX(b[0], b[1], b[2]);
		EulerFactorYZX(eigen_yzx, c[0], c[1], c[2]);
		assert(std::abs(b[0] - c[0]) < 1e-12 && std::abs(b[1] - c[1]) < 1e-12 && std::abs(b[2] - c[2]) < 1e-12);

		Eigen::Matrix3d eigen_zxy = EulerMakerZXY(a[0], a[1], a[2]);
		Wm5::Matrix3d wm5_zxy;
		wm5_zxy.MakeEulerZXY(a[0], a[1], a[2]);
		std::cout << "Check ZXY" << a[0] << " " << a[1] << " " << a[2] << std::endl;
		check(eigen_zxy, wm5_zxy);
		wm5_zxy.ExtractEulerZXY(b[0], b[1], b[2]);
		EulerFactorZXY(eigen_zxy, c[0], c[1], c[2]);
		assert(std::abs(b[0] - c[0]) < 1e-12 && std::abs(b[1] - c[1]) < 1e-12 && std::abs(b[2] - c[2]) < 1e-12);

		Eigen::Matrix3d eigen_zyx = EulerMakerZYX(a[0], a[1], a[2]);
		Wm5::Matrix3d wm5_zyx;
		wm5_zyx.MakeEulerZYX(a[0], a[1], a[2]);
		std::cout << "Check ZYX" << a[0] << " " << a[1] << " " << a[2] << std::endl;
		check(eigen_zyx, wm5_zyx);
		wm5_zyx.ExtractEulerZYX(b[0], b[1], b[2]);
		EulerFactorZYX(eigen_zyx, c[0], c[1], c[2]);
		//std::cout << "b[0] = " << b[0] << "  b[1] = " << b[1] << "  b[2] = " << b[2] << std::endl;
		//std::cout << "c[0] = " << c[0] << "  c[1] = " << c[1] << "  c[2] = " << c[2] << std::endl;
		assert(std::abs(b[0] - c[0]) < 1e-12 && std::abs(b[1] - c[1]) < 1e-12 && std::abs(b[2] - c[2]) < 1e-12);
	}
	return 0;
}

int main()
{
	Eigen::Matrix3d a, b;
	a << 1, 2, 4, 8, 16, 32, 128, 256, 512;
	std::cout << a << std::endl;

	b = Eigen::Matrix3d::Zero();
	int t = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			b(i, j) = ++t;
		}
	}
	std::cout << b << std::endl;

	std::cout << a * b << std::endl; // 看起来行优先或是列优先不影响最后计算的结果...
	// Wm5是行优先的, 但是Eigen是列优先的...

	return 0;
}