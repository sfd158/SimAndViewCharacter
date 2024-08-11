// Borrow some come from AI4Animation

#include <Mat3Euler.h>
#include "Common.h"

extern "C"
{
	// ===========MatrixXd==================================
    EXPORT_API MatrixXd* MatrixXdCreate(int rows, int cols) 
	{
		MatrixXd* res = new MatrixXd(rows, cols);
		res->setZero();
		return res;
	}

	EXPORT_API MatrixXf* MatrixXd_to_f(const MatrixXd* ptr)
	{
		MatrixXf* res = new MatrixXf(ptr->rows(), ptr->cols());
		size_t size = ptr->size();
		for (size_t i = 0; i < size; i++) 
			res->data()[i] = static_cast<float>(ptr->data()[i]);
		return res;
	}

	EXPORT_API MatrixXd* MatrixXf_to_d(const MatrixXd* ptr)
	{
		MatrixXd* res = new MatrixXd(ptr->rows(), ptr->cols());
		size_t size = ptr->size();
		for (size_t i = 0; i < size; i++)
			res->data()[i] = ptr->data()[i];
		return res;
	}

	EXPORT_API MatrixXd* MatrixXdCreateGaussian(int rows, int cols)
	{
		MatrixXd* mat = new MatrixXd(rows, cols);
		static std::default_random_engine generator;
		static std::normal_distribution<double> distribution(0.0f, 1.0); // Mean 0, Stddev 1

		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				(*mat)(i, j) = distribution(generator);

		return mat;
	}

	EXPORT_API void MatrixXdDelete(MatrixXd* ptr) 
	{
		if (ptr != nullptr)
			delete(ptr);
	}

	EXPORT_API int MatrixXdGetRows(MatrixXd* ptr) 
	{
		return static_cast<int>((*ptr).rows());
	}

	EXPORT_API int MatrixXdGetCols(MatrixXd* ptr) 
	{
		return static_cast<int>((*ptr).cols());
	}

	EXPORT_API void MatrixXdSetZero(MatrixXd* ptr) 
	{
		ptr->setZero();
	}

	EXPORT_API void MatrixXdSetSize(MatrixXd* ptr, int rows, int cols)
	{
		(*ptr).conservativeResize(rows, cols);
	}

	EXPORT_API void MatrixXdAdd(MatrixXd* lhs, MatrixXd* rhs, MatrixXd* out) 
	{
		(*out).noalias() = *lhs + *rhs;
	}

	EXPORT_API void MatrixXdSubtract(MatrixXd* lhs, MatrixXd* rhs, MatrixXd* out) 
	{
		(*out).noalias() = *lhs - *rhs;
	}

	EXPORT_API void MatrixXdProduct(MatrixXd* lhs, MatrixXd* rhs, MatrixXd* out) 
	{
		(*out).noalias() = (*lhs) * (*rhs);
	}

	EXPORT_API void MatrixXdScale(MatrixXd* lhs, double value, MatrixXd* out)
	{
		(*out).noalias() = *lhs * value;
	}

	EXPORT_API void MatrixXdSetValue(MatrixXd* ptr, int row, int col, double value) 
	{
		(*ptr)(row, col) = value;
	}

	EXPORT_API double MatrixXdGetValue(MatrixXd* ptr, int row, int col) 
	{
		return (*ptr)(row, col);
	}

	EXPORT_API void MatrixXdConcat(MatrixXd* a, MatrixXd* b, MatrixXd* res, int axis)
	{
		eigen_assert(axis == 0 || axis == 1);
		size_t ac = a->cols(), ar = a->rows(), bc = b->cols(), br = b->rows(), rc = a->cols(), rr = b->rows();
		if (axis == 0)
		{
			eigen_assert(rr == ar + br && rc == ac && rc == bc);
			res->block(0, 0, ar, ac) = *a;
			res->block(ar, 0, br, bc) = *b;
		}
		else if (axis == 1)
		{
			eigen_assert(rc == ac + bc && rr == ar && rr == br);
			res->block(0, 0, ar, ac) = *a;
			res->block(0, ac, br, bc) = *b;
		}
	}

	// =====================================================

	// ===============Matrix3d==============================
	EXPORT_API Matrix3d* Matrix3dCreate() 
	{
		return new Matrix3d();
	}

	EXPORT_API Matrix3d * Matrix3dCreateValue(
		double a00, double a01, double a02,
		double a10, double a11, double a12,
		double a20, double a21, double a22)
	{
		Matrix3d * res = new Matrix3d();
		(*res)(0, 0) = a00;
		(*res)(0, 1) = a01;
		(*res)(0, 2) = a02;
		(*res)(1, 0) = a10;
		(*res)(1, 1) = a11;
		(*res)(1, 2) = a12;
		(*res)(2, 0) = a20;
		(*res)(2, 1) = a21;
		(*res)(2, 2) = a22;
		return res;
	}

	EXPORT_API void Matrix3dDelete(Matrix3d* ptr) 
	{
		delete ptr;
	}

	EXPORT_API void Matrix3dSetValue(Matrix3d* ptr, int row, int col, double value) 
	{
		(*ptr)(row, col) = value;
	}

	EXPORT_API double Matrix3dGetValue(Matrix3d* ptr, int row, int col)
	{
		return (*ptr)(row, col);
	}

	EXPORT_API void Matrix3dProduct(Matrix3d* lhs, Matrix3d* rhs, Matrix3d* out) 
	{
		(*out).noalias() = (*lhs) * (*rhs);
	}

	EXPORT_API void Matrix3dAdd(Matrix3d* lhs, Matrix3d* rhs, Matrix3d* out)
	{
		(*out).noalias() = (*lhs) + (*rhs);
	}

	EXPORT_API void Matrix3dMinus(Matrix3d* lhs, Matrix3d* rhs, Matrix3d* out)
	{
		(*out).noalias() = (*lhs) - (*rhs);
	}

	EXPORT_API void Matrix3dScale(Matrix3d* lhs, double value, MatrixXd* out)
	{
		(*out).noalias() = *lhs * value;
	}

	EXPORT_API void Mat3fProductVec3f(Matrix3d * lhs, Vector3d * rhs, Vector3d* out)
	{
		(*out).noalias() = (*lhs) * (*rhs);
	}

	EXPORT_API double Matrix3dDet(Matrix3d* lhs)
	{
		return (*lhs).determinant();
	}

	EXPORT_API void Mat3fProductVec3fPointer(Matrix3d * lhs, double x, double y, double z, double * out)
	{
		Vector3d rhs(x, y, z);
		Vector3d res = (*lhs) * rhs;
		out[0] = res.x(); out[1] = res.y(); out[2] = res.z();
	}

	// ===============Quaternion Maker======================
	EXPORT_API Matrix3d* Matrix3dFromQuaternion(double x, double y, double z, double w) 
	{
		Quaterniond q(w, x, y, z);
		Matrix3d* m = new Matrix3d(q.toRotationMatrix());
		return m;
	}

	// ===============Quaternion Extractor==================
	EXPORT_API QuaternionResultd Matrix3dToQuaternion(Matrix3d* ptr)
	{
		Quaterniond q(*ptr);
		return {q.x(), q.y(), q.z(), q.w()};
	} 

	// ===============Euler Maker===========================
	EXPORT_API Matrix3d* Matrix3dMakeEulerXYZ(double xAngle, double yAngle, double zAngle)
	{
		return Matrix3MakeEulerXYZ<double>(xAngle, yAngle, zAngle);
	}

	EXPORT_API Matrix3d* Matrix3dMakeEulerXZY(double xAngle, double yAngle, double zAngle)
	{
		return Matrix3MakeEulerXZY<double>(xAngle, zAngle, yAngle);
	}

	EXPORT_API Matrix3d* Matrix3dMakeEulerYXZ(double xAngle, double yAngle, double zAngle)
	{
		return Matrix3MakeEulerYXZ<double>(yAngle, xAngle, zAngle);
	}

	EXPORT_API Matrix3d* Matrix3dMakeEulerYZX(double xAngle, double yAngle, double zAngle)
	{
		return Matrix3MakeEulerYZX<double>(yAngle, zAngle, xAngle);
	}

	EXPORT_API Matrix3d* Matrix3dMakeEulerZXY(double xAngle, double yAngle, double zAngle)
	{
		return Matrix3MakeEulerZXY<double>(zAngle, xAngle, yAngle);
	}

	EXPORT_API Matrix3d* Matrix3dMakeEulerZYX(double xAngle, double yAngle, double zAngle)
	{
		return Matrix3MakeEulerZYX<double>(zAngle, yAngle, xAngle);
	}

	// =============Euler Extractor=========================

	EXPORT_API EulerResultd Matrix3dExtractEulerXYZ(Matrix3d* ptr)
	{
		return Matrix3ExtractEulerXYZ<double, EulerResultd>(ptr);
	}

	EXPORT_API EulerResultd Matrix3dExtractEulerXZY(Matrix3d* ptr) 
	{
		return Matrix3ExtractEulerXZY<double, EulerResultd>(ptr);
	}

	EXPORT_API EulerResultd Matrix3dExtractEulerYXZ(Matrix3d* ptr)
	{
		return Matrix3ExtractEulerYXZ<double, EulerResultd>(ptr);
	}

	EXPORT_API EulerResultd Matrix3dExtractEulerYZX(Matrix3d* ptr)
	{
		return Matrix3ExtractEulerYZX<double, EulerResultd>(ptr);
	}

	EXPORT_API EulerResultd Matrix3dExtractEulerZXY(Matrix3d* ptr)
	{
		return Matrix3ExtractEulerZXY<double, EulerResultd>(ptr);
	}

	EXPORT_API EulerResultd Matrix3dExtractEulerZYX(Matrix3d* ptr)
	{
		return Matrix3ExtractEulerZYX<double, EulerResultd>(ptr);
	}
}
