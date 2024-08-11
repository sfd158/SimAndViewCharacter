// C language Wrapper of inverse dynamics
#pragma once
#include <inversedynamics.h>


//#ifdef __cplusplus
//extern "C" {
//#endif

	CInverseDynamicsPtr InvDynCreate(
		const std::vector<double>* body_mass,
		const std_vector_Matrix3d* body_inertia,
		const std_vector_Vector3d* body_position,
		const std_vector_Matrix3d* body_rotation,
		const std::vector<int>* parent_joint_dof,
		const std_vector_Vector3d* parent_joint_pos,
		const std::vector<std::string>* parent_joint_euler_order,
		const std_vector_Matrix3d* parent_joint_euler_axis,
		const std::vector<int>* parent_body_index);

	int InvDynGeneralizedCoordinatesDimension(CInverseDynamicsPtr ptr);
	void InvDynConvertToGeneralizeCoordinates(
		CInverseDynamicsPtr ptr,
		const Eigen_Vector3d* rootPos, // In
		const std_vector_Quaterniond* vJointRots, // In
		std::vector<double>* q // Out
	);

	void InvDynConvertToGeneralizeCoordinatesBatch(
		CInverseDynamicsPtr ptr,
		const std_vector_Vector3d* rootPosBatch,
		const std_vector_std_vector_Quaterniond* vJointRotsBatch,
		std::vector<std::vector<double> >* q
	);

	void InvDynConvertToJointRotations(
		CInverseDynamicsPtr ptr,
		const std::vector<double>* q,
		Eigen_Vector3d* rootPos,
		std_vector_Quaterniond* vJointRots
	);

	void InvDynInitializeInverseDynamics(
		CInverseDynamicsPtr ptr,
		const std::vector<double>* q,
		const std::vector<double>* qdot,
		const std::vector<double>* qdotdot
	);

	void InvDynComputeVelocityAcceleration(CInverseDynamicsPtr ptr, double gravity_x, double gravity_y, double gravity_z);
	
	void InvDynComputeForceTorqueBatch(
		CInverseDynamicsPtr ptr,
		const std::vector<std::vector<double> >* q,
		const std::vector<std::vector<double> >* dq,
		const std::vector<std::vector<double> >* ddq,
		const Eigen_Vector3d* gravity,
		std::vector<std_vector_Vector3d>* linvels, // Out
		std::vector<std_vector_Vector3d>* angvels, // Out
		std::vector<std_vector_Vector3d>* linaccs, // Out
		std::vector<std_vector_Vector3d>* angaccs, // Out
		std::vector<std_vector_Vector3d>* fs_in, // In
		std::vector<std_vector_Vector3d>* ts_in, // In
		std::vector<std_vector_Vector3d>* f_local, // Out
		std::vector<std_vector_Vector3d>* t_local, // Out
		std::vector<double>* com_mass, // Out
		std_vector_Vector3d* com_pos, // Out
		std_vector_Vector3d* com_linvel, // Out
		std_vector_Vector3d* com_ang_momentums, // Out
		std_vector_Matrix3d* com_inertia // Out
	);

	void InvDynGetLocalLinearVelocity(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);
	void InvDynGetLocalAngularVelocity(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);
	void InvDynGetLocalLinearAcceleration(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);
	void InvDynGetLocalAngularAcceleration(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);
	void InvDynComputeForceTorque(CInverseDynamicsPtr ptr, std_vector_Vector3d* force, std_vector_Vector3d* torque);

	void InvDynGetLocalForce(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);
	void InvDynGetLocalTorque(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);
	void InvDynGetJointRotation(CInverseDynamicsPtr ptr, std_vector_Matrix3d_ptr* out);
	void InvDynGetBodyOrientation(const CInverseDynamicsPtr ptr, std_vector_Matrix3d_ptr* out);
	
	int InvDygGetTotalJointDof(const CInverseDynamicsPtr ptr);
	int InvDynGetJointDof(const CInverseDynamicsPtr ptr, int bid);
	void InvDynReRoot(CInverseDynamicsPtr ptr, int bodyId);
	void InvDynReComputeR(CInverseDynamicsPtr ptr);
	void InvDynClearReRootFlag(CInverseDynamicsPtr ptr);
	void InvDynComputeBodyPositions(CInverseDynamicsPtr ptr, std_vector_Vector3d* out);
	void InvDynComputeComMomentums(CInverseDynamicsPtr ptr, double* Mass, Eigen_Vector3d* com,
		Eigen_Vector3d* linVelocity, Eigen_Vector3d* angMomentum, Eigen_Matrix3d* inertia);
	void InvDynComputeLocalW_q(const CInverseDynamicsPtr ptr, int bodyId, std_vector_Vector3d* out);
	void InvDynGetCls(CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);
	void InvDynGetDCls(CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out);

	void DestroyInverseDynamics(CInverseDynamicsPtr ptr);
	void DestroyInverseDynamicsHandle(size_t ptr);


//#ifdef __cplusplus
//}
//#endif

