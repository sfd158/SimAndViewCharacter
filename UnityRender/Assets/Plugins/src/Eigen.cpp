// Borrow some come from AI4Animation

#include <Mat3Euler.h>
#include "Common.h"

extern "C"
{
	// ===========MatrixXf==================================
    EXPORT_API MatrixXf* MatrixXfCreate(int rows, int cols) 
	{
		MatrixXf* res = new MatrixXf(rows, cols);
		res->setZero();
		return res;
	}

	EXPORT_API MatrixXf* MatrixXfCreateGaussian(int rows, int cols)
	{
		MatrixXf* mat = new MatrixXf(rows, cols);
		static std::default_random_engine generator;
		static std::normal_distribution<float> distribution(0.0f, 1.0f); // Mean 0, Stddev 1

		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				(*mat)(i, j) = distribution(generator);

		return mat;
	}

	EXPORT_API void MatrixXfDelete(MatrixXf* ptr) 
	{
		if (ptr != nullptr)
			delete(ptr);
	}

	EXPORT_API int MatrixXfGetRows(MatrixXf* ptr) 
	{
		return static_cast<int>((*ptr).rows());
	}

	EXPORT_API int MatrixXfGetCols(MatrixXf* ptr) 
	{
		return static_cast<int>((*ptr).cols());
	}

	EXPORT_API void MatrixXfSetZero(MatrixXf* ptr) 
	{
		ptr->setZero();
	}

	EXPORT_API void MatrixXfSetSize(MatrixXf* ptr, int rows, int cols)
	{
		(*ptr).conservativeResize(rows, cols);
	}

	EXPORT_API void MatrixXfAdd(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) 
	{
		(*out).noalias() = *lhs + *rhs;
	}

	EXPORT_API void MatrixXfSubtract(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) 
	{
		(*out).noalias() = *lhs - *rhs;
	}

	EXPORT_API void MatrixXfProduct(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) 
	{
		(*out).noalias() = (*lhs) * (*rhs);
	}

	EXPORT_API void MatrixXfScale(MatrixXf* lhs, float value, MatrixXf* out)
	{
		(*out).noalias() = *lhs * value;
	}

	EXPORT_API void MatrixXfSetValue(MatrixXf* ptr, int row, int col, float value) 
	{
		(*ptr)(row, col) = value;
	}

	EXPORT_API float MatrixXfGetValue(MatrixXf* ptr, int row, int col) 
	{
		return (*ptr)(row, col);
	}

	EXPORT_API void MatrixXfConcat(MatrixXf* a, MatrixXf* b, MatrixXf* res, int axis)
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

	EXPORT_API void RELU(MatrixXf* ptr) 
	{
		(*ptr).noalias() = (*ptr).cwiseMax(0.0f);
	}

	EXPORT_API void ELU(MatrixXf* ptr)
	{
		(*ptr).noalias() = ((*ptr).array().cwiseMax(0.0f) + (*ptr).array().cwiseMin(0.0f).exp() - 1.0f).matrix();
	}

	EXPORT_API void Sigmoid(MatrixXf* ptr) {
		size_t rows = (*ptr).rows();
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) = 1.0f / (1.0f + std::exp(-(*ptr)(i, 0)));
		}
	}

	EXPORT_API void TanH(MatrixXf* ptr) {
		size_t rows = (*ptr).rows();
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) = std::tanh((*ptr)(i, 0));
		}
	}

	EXPORT_API void SoftMax(MatrixXf* ptr) {
		float frac = 0.0f;
		size_t rows = (*ptr).rows();
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) = std::exp((*ptr)(i, 0));
			frac += (*ptr)(i, 0);
		}
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) /= frac;
		}
	}

	EXPORT_API void LogSoftMax(MatrixXf* ptr) {
		float frac = 0.0f;
		size_t rows = (*ptr).rows();
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) = std::exp((*ptr)(i, 0));
			frac += (*ptr)(i, 0);
		}
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) = std::log((*ptr)(i, 0) / frac);
		}
	}

	EXPORT_API void SoftSign(MatrixXf* ptr) {
		size_t rows = (*ptr).rows();
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) /= 1 + std::abs((*ptr)(i, 0));
		}
	}

	EXPORT_API void Exp(MatrixXf* ptr) {
		size_t rows = (*ptr).rows();
		for (size_t i = 0; i < rows; i++) {
			(*ptr)(i, 0) = std::exp((*ptr)(i, 0));
		}
	}

	EXPORT_API void PointwiseProduct(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out)
	{
		(*out).noalias() = (*lhs).cwiseProduct(*rhs);
	}

	EXPORT_API void PointwiseQuotient(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out)
	{
		(*out).noalias() = (*lhs).cwiseQuotient(*rhs);
	}

	EXPORT_API void PointwiseAbsolute(MatrixXf* in, MatrixXf* out)
	{
		(*out).noalias() = (*in).cwiseAbs();
	}

	EXPORT_API float RowSum(MatrixXf* ptr, int row)
	{
		return (*ptr).row(row).sum();
	}

	EXPORT_API float ColSum(MatrixXf* ptr, int col)
	{
		return (*ptr).col(col).sum();
	}

	EXPORT_API float RowMean(MatrixXf* ptr, int row)
	{
		return (*ptr).row(row).mean();
	}

	EXPORT_API float ColMean(MatrixXf* ptr, int col)
	{
		return (*ptr).col(col).mean();
	}

	EXPORT_API float RowStd(MatrixXf* ptr, int row)
	{
		MatrixXf diff = (*ptr).row(row) - (*ptr).row(row).mean() * MatrixXf::Ones(1, (*ptr).rows());
		diff = diff.cwiseProduct(diff);
		return std::sqrt(diff.sum() / (*ptr).cols());
	}

	EXPORT_API float ColStd(MatrixXf* ptr, int col)
	{
		MatrixXf diff = (*ptr).col(col) - (*ptr).col(col).mean() * MatrixXf::Ones((*ptr).rows(), 1);
		diff = diff.cwiseProduct(diff);
		return std::sqrt(diff.sum() / (*ptr).rows());
	}

	EXPORT_API void Normalise(MatrixXf* ptr, MatrixXf* mean, MatrixXf* std, MatrixXf* out)
	{
		(*out).noalias() = (*ptr - *mean).cwiseQuotient(*std);
	}

	EXPORT_API void Renormalise(MatrixXf* ptr, MatrixXf* mean, MatrixXf* std, MatrixXf* out) 
	{
		(*out).noalias() = (*ptr).cwiseProduct(*std) + *mean;
	}

	EXPORT_API void Layer(MatrixXf* in, MatrixXf* W, MatrixXf* b, MatrixXf* out)
	{
		(*out).noalias() = *W * *in + *b;
	}

	// =====================================================

	// ===============Matrix3f==============================
	EXPORT_API Matrix3f* Matrix3fCreate() 
	{
		return new Matrix3f();
	}

	EXPORT_API Matrix3f * Matrix3fCreateValue(
		float a00, float a01, float a02,
		float a10, float a11, float a12,
		float a20, float a21, float a22)
	{
		Matrix3f * res = new Matrix3f();
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

	EXPORT_API void Matrix3fDelete(Matrix3f* ptr) 
	{
		delete ptr;
	}

	EXPORT_API void Matrix3fSetValue(Matrix3f* ptr, int row, int col, float value) 
	{
		(*ptr)(row, col) = value;
	}

	EXPORT_API float Matrix3fGetValue(Matrix3f* ptr, int row, int col)
	{
		return (*ptr)(row, col);
	}

	EXPORT_API void Matrix3fProduct(Matrix3f* lhs, Matrix3f* rhs, Matrix3f* out) 
	{
		(*out).noalias() = (*lhs) * (*rhs);
	}

	EXPORT_API void Matrix3fAdd(Matrix3f* lhs, Matrix3f* rhs, Matrix3f* out)
	{
		(*out).noalias() = (*lhs) + (*rhs);
	}

	EXPORT_API void Matrix3fMinus(Matrix3f* lhs, Matrix3f* rhs, Matrix3f* out)
	{
		(*out).noalias() = (*lhs) - (*rhs);
	}

	EXPORT_API void Matrix3fScale(Matrix3f* lhs, float value, MatrixXf* out)
	{
		(*out).noalias() = *lhs * value;
	}

	EXPORT_API void Mat3fProductVec3f(Matrix3f * lhs, Vector3f * rhs, Vector3f* out)
	{
		(*out).noalias() = (*lhs) * (*rhs);
	}

	EXPORT_API float Matrix3fDet(Matrix3f* lhs)
	{
		return (*lhs).determinant();
	}

	EXPORT_API void Mat3fProductVec3fPointer(Matrix3f * lhs, float x, float y, float z, float * out)
	{
		Vector3f rhs(x, y, z);
		Vector3f res = (*lhs) * rhs;
		out[0] = res.x(); out[1] = res.y(); out[2] = res.z();
	}

	// ===============Quaternion Maker======================
	EXPORT_API Matrix3f* Matrix3fFromQuaternion(float x, float y, float z, float w) 
	{
		Quaternionf q(w, x, y, z);
		Matrix3f* m = new Matrix3f(q.toRotationMatrix());
		return m;
	}

	// ===============Quaternion Extractor==================
	EXPORT_API QuaternionResultf Matrix3fToQuaternion(Matrix3f* ptr)
	{
		Quaternionf q(*ptr);
		return {q.x(), q.y(), q.z(), q.w()};
	} 

	// ===============Euler Maker===========================
	EXPORT_API Matrix3f* Matrix3fMakeEulerXYZ(float xAngle, float yAngle, float zAngle)
	{
		return Matrix3MakeEulerXYZ<float>(xAngle, yAngle, zAngle);
	}

	EXPORT_API Matrix3f* Matrix3fMakeEulerXZY(float xAngle, float yAngle, float zAngle)
	{
		return Matrix3MakeEulerXZY<float>(xAngle, zAngle, yAngle);
	}

	EXPORT_API Matrix3f* Matrix3fMakeEulerYXZ(float xAngle, float yAngle, float zAngle)
	{
		return Matrix3MakeEulerYXZ<float>(yAngle, xAngle, zAngle);
	}

	EXPORT_API Matrix3f* Matrix3fMakeEulerYZX(float xAngle, float yAngle, float zAngle)
	{
		return Matrix3MakeEulerYZX<float>(yAngle, zAngle, xAngle);
	}

	EXPORT_API Matrix3f* Matrix3fMakeEulerZXY(float xAngle, float yAngle, float zAngle)
	{
		return Matrix3MakeEulerZXY<float>(zAngle, xAngle, yAngle);
	}

	EXPORT_API Matrix3f* Matrix3fMakeEulerZYX(float xAngle, float yAngle, float zAngle)
	{
		return Matrix3MakeEulerZYX<float>(zAngle, yAngle, xAngle);
	}

	// =============Euler Extractor=========================

	EXPORT_API EulerResultf Matrix3fExtractEulerXYZ(Matrix3f* ptr)
	{
		return Matrix3ExtractEulerXYZ<float, EulerResultf>(ptr);
	}

	EXPORT_API EulerResultf Matrix3fExtractEulerXZY(Matrix3f* ptr) 
	{
		return Matrix3ExtractEulerXZY<float, EulerResultf>(ptr);
	}

	EXPORT_API EulerResultf Matrix3fExtractEulerYXZ(Matrix3f* ptr)
	{
		return Matrix3ExtractEulerYXZ<float, EulerResultf>(ptr);
	}

	EXPORT_API EulerResultf Matrix3fExtractEulerYZX(Matrix3f* ptr)
	{
		return Matrix3ExtractEulerYZX<float, EulerResultf>(ptr);
	}

	EXPORT_API EulerResultf Matrix3fExtractEulerZXY(Matrix3f* ptr)
	{
		return Matrix3ExtractEulerZXY<float, EulerResultf>(ptr);
	}

	EXPORT_API EulerResultf Matrix3fExtractEulerZYX(Matrix3f* ptr)
	{
		return Matrix3ExtractEulerZYX<float, EulerResultf>(ptr);
	}

	EXPORT_API Matrix4f * Matrix4fCreate()
	{
		return new Matrix4f();
	}

	EXPORT_API void Matrix4fDelete(Matrix3f* ptr)
	{
		if (ptr != nullptr)
		{
			delete ptr;
		}
	}

}
