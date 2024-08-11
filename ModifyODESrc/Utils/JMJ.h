#pragma once
#include <Eigen/Dense>
#include <EigenExtension/EigenBindingWrapper.h>
using Matrix66 = Eigen::Matrix<double, 6, 6>;
using Matrixdd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Matrix33 = Eigen::Matrix3d;
//#define DEBUGJMJ

namespace JMJ
{
	class JMJBase {
	public:
		int body_num, ball_num, hinge_num;
		std::vector<int> parent, order2idx, jointchild;
		std::vector<double> mass, joint_d;
		double dt;
	
		void Reset(const int _body_num, const double* _mass, const int _ball_num, const int _hinge_num,
			const int* _bidx0, const int* _bidx1, const double* kds, const double* cfm, const double _dt);
	};

	struct treeNodeBase
	{
		Matrix66 D, J;
		Eigen::LDLT<Matrix66> Dinv;
		int parent;
		int idx;
		bool isConstraint;
		treeNodeBase() {
			D.setZero(); J.setZero(); 
			parent= -1;
			idx = -1;
		}

		void set_Parent(treeNodeBase* parentPointer) {
			this->parent = parentPointer->idx;
		}

	};

	struct bodyNode : public treeNodeBase
	{

		void setMass(const double& mass,const double* inertia)
		{
			isConstraint = false;
			D.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * mass;
			if (inertia == NULL) return;
			D.block(3, 3, 3, 3) = Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> (inertia);
		}

		bodyNode(const double& mass = 0.0,const double* inertia = NULL)
		{
			setMass(mass, inertia);
			isConstraint = false;
		}
	};

	struct jointNode : public treeNodeBase
	{
		int m;
		jointNode() { isConstraint = true; }
		void set_ball(const double* _Jlin0, const double *_Jlin1,
			const double *_Jang0, const double *_Jang1,
			const double *R,
			const double *kds, const double cfm, const double dt,
			treeNodeBase* childPointer
		)
		{
			m = 3;
			isConstraint = true;
			// set D
			for (int i = 0; i < 3; i++) D(i, i) = -cfm / dt;
			for (int i = 0; i < 3; i++) D(i + 3, i + 3) = -1.0 / (dt * kds[i]);

			// set J for self
			J.block(0, 0, 3, 3) = -(Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> (_Jlin1));
			J.block(0, 3, 3, 3) = -(Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> (_Jang1));
			J.block(3, 3, 3, 3) = - (Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> (R)).transpose();

			// set J for child
			childPointer->J.block(0, 0, 3, 3) = -(Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> (_Jlin0));
			childPointer->J.block(0, 3, 3, 3) = -(Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> (_Jang0));
			childPointer->J.block(3, 3, 3, 3) = (Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> (R)).transpose();
			childPointer->J.transposeInPlace();
		}


		void set_hinge(const double* _Jlin0, const double *_Jlin1,
			const double *_Jang0, const double *_Jang1,
			const double *_axis,
			const double *kds, const double cfm, const double dt,
			treeNodeBase* childPointer
		)
		{
			m = 5;
			isConstraint = true;
			// set D
			for (int i = 0; i < 5; i++) D(i, i) = -cfm / dt;
			D(5, 5) = - 1.0 / (dt * kds[0]);

			// set J for self
			J.block(0, 0, 5, 3) = -(Eigen::Map<const Eigen::Matrix<double,5,3,Eigen::RowMajor>> (_Jlin1));
			J.block(0, 3, 5, 3) = -(Eigen::Map<const Eigen::Matrix<double,5,3,Eigen::RowMajor>> (_Jang1));
			J.block(5, 3, 1, 3) = - (Eigen::Map<const Eigen::Matrix<double,1,3,Eigen::RowMajor>> (_axis));

			// set J for child
			childPointer->J.block(0, 0, 5, 3) = -(Eigen::Map<const Eigen::Matrix<double,5,3,Eigen::RowMajor>> (_Jlin0));
			childPointer->J.block(0, 3, 5, 3) = -(Eigen::Map<const Eigen::Matrix<double,5,3,Eigen::RowMajor>> (_Jang0));
			childPointer->J.block(5, 3, 1, 3) = (Eigen::Map<const Eigen::Matrix<double,1,3,Eigen::RowMajor>> (_axis));

			childPointer->J.transposeInPlace();
		}

		void set_jacobian_fast(int _m, const double* J0, const double* J1, const double* dJ, const double* _D, treeNodeBase* childPointer) {
			m = _m;
			isConstraint = true;
			for (int i = 0; i < 6; i++) D(i, i) = _D[i];
			J.block(0, 0, m, 6) = -Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(J1, m, 6);
			J.block(m, 3, 6-m, 3) = -Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dJ, 3, 6-m).transpose();

			childPointer->J.block(0, 0, m, 6) = -Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(J0, m, 6);
			childPointer->J.block(m, 3, 6-m, 3) = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(dJ, 3, 6-m).transpose();
			childPointer->J.transposeInPlace();
		}
	};




	class JMinvJ
	{
		/*
		 * this class calculate J(M^{-1})Jx=b
		 * J should encode connection of rigid body and it must be a tree
		 * (which means every node can have only ONE parent)
		 */

		int num_node;
		int num_body;
		std::vector<int> parent;
		std::vector<treeNodeBase*> myNode;
		std::vector<int> order2Idx;

		void factorJMJ();

		void buildBeforeSolve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x);
#ifdef DEBUGJMJ
		Eigen::MatrixXd A;
#endif // DEBUGJMJ

	public:

		void _Reset(
			const int _num_body, const double* mass,const double* Inertia,
			const int num_joint,
			const double* Jlin0, const double* Jlin1,
			const double* Jang0, const double* Jang1,
			const int* bidx0, const int* bidx1,
			const double* R, const double* axis,
			const int* hinge_flag, const double* kds,
			const double* cfm, const double* c, const double dt
		);
		
		void _ResetWithBase(
			const double* Inertia, const double* JB0, const double* JB1, const double* JH0, const double* JH1,
			const double* R, const double* axis, const JMJBase* _base
		);

		JMinvJ() {};
		
		void solveX(double* _b, double* _lambda, const int col);

		void backward_solve(double* b, double* lambda, double* G_b, double* G_Lambda, const int col, double* G_I, double* G_Jl0, double* G_Jl1, double* G_Ja0, double* G_Ja1, 
                    double* G_R, double* G_axis);
	};


}


class JMinvJCWrapper {

	JMJ::JMinvJ solver;

public:
	JMinvJCWrapper() {};
	void Reset(
			const int _num_body, const double* mass,const double* Inertia,
			const int num_joint,
			const double* Jlin0, const double* Jlin1,
			const double* Jang0, const double* Jang1,
			const int* bidx0, const int* bidx1,
			const double* R, const double* axis,
			const int* hinge_flag, const double* kds,
			const double* cfm, const double* c, const double dt) {

		solver._Reset(_num_body,mass,Inertia, num_joint, Jlin0,Jlin1,
		 Jang0,Jang1, bidx0,bidx1, R,axis, hinge_flag,kds, cfm,c,dt);

	}
	
	void Reset_base(const double* Inertia,
		const double* JB0,
		const double* JB1,
		const double* JH0,
		const double* JH1,
		const double* R, const double* axis,
		const JMJ::JMJBase* _base) {
		solver._ResetWithBase(Inertia, JB0, JB1, JH0, JH1, R, axis, _base);
	}
	void solve(double* _b, double* _lambda, const int col){
		solver.solveX(_b, _lambda, col);
	}
	void backward_solve(double* b, double* lambda, double* G_b, double* G_Lambda, const int col, double* G_I, double* G_Jl0, double* G_Jl1, double* G_Ja0, double* G_Ja1, 
                    double* G_R, double* G_axis
						){
		solver.backward_solve(b,lambda,G_b,G_Lambda,col,G_I,G_Jl0,G_Jl1,G_Ja0,G_Ja1,G_R,G_axis);		
	}
};