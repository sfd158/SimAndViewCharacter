#include <inversedynamicswrapper.h>
#include <EigenExtension/EigenBindingWrapper.h>


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
		const std::vector<int>* parent_body_index)
	{
		CInverseDynamicsPtr ptr = new CInverseDynamics(
			*body_mass,
			body_inertia->data,
			body_position->data,
			body_rotation->data,
			*parent_joint_dof,
			parent_joint_pos->data,
			*parent_joint_euler_order,
			parent_joint_euler_axis->data,
			*parent_body_index);

		return ptr;
	}

	void CalcInverseDynamics()
	{

	}

	int InvDynGeneralizedCoordinatesDimension(CInverseDynamicsPtr ptr)
	{
		return ptr->GeneralizedCoordinatesDimension();
	}

	void InvDynConvertToGeneralizeCoordinates(
		CInverseDynamicsPtr ptr, 
		const Eigen_Vector3d * rootPos, // In
		const std_vector_Quaterniond * vJointRots, // In
		std::vector<double> * q
		)
	{
		ptr->ConvertToGeneralizedCoordinates(rootPos->data, vJointRots->data, *q);
	}

	void InvDynConvertToGeneralizeCoordinatesBatch(
		CInverseDynamicsPtr ptr,
		const std_vector_Vector3d * rootPosBatch,
		const std_vector_std_vector_Quaterniond * vJointRotsBatch,
		std::vector<std::vector<double> > * q
	)
	{
		dCASSERT(rootPosBatch->size() == vJointRotsBatch->size_0());
		size_t Size = rootPosBatch->size();
		q->resize(Size);
		for (size_t i = 0; i < Size; i++)
		{
			ptr->ConvertToGeneralizedCoordinates(rootPosBatch->data[i], vJointRotsBatch->data[i], (*q)[i]);
		}
	}

	void InvDynConvertToJointRotations(
		CInverseDynamicsPtr ptr, 
		const std::vector<double> * q,
		Eigen_Vector3d * rootPos,
		std_vector_Quaterniond * vJointRots)
	{
		ptr->ConvertToJointRotations(*q, rootPos->data, vJointRots->data);
	}

	void InvDynInitializeInverseDynamics(
		CInverseDynamicsPtr ptr,
		const std::vector<double> * q, 
		const std::vector<double> * qdot,
		const std::vector<double> * qdotdot
	)
	{
		ptr->InitializeInverseDynamics(*q, *qdot, *qdotdot);
	}

	void InvDynComputeVelocityAcceleration(CInverseDynamicsPtr ptr, double gravity_x, double gravity_y, double gravity_z)
	{
		Vector3d g = Vector3d::Zero();
		g << gravity_x, gravity_y, gravity_z;
		ptr->ComputeVelocityAcceleration(g);
	}

	void InvDynComputeForceTorqueBatch(
		CInverseDynamicsPtr ptr,
		const std::vector<std::vector<double> > * q,
		const std::vector<std::vector<double> > * dq,
		const std::vector<std::vector<double> > * ddq,
		const Eigen_Vector3d * gravity,
		std::vector<std_vector_Vector3d> * linvels, // Out
		std::vector<std_vector_Vector3d> * angvels, // Out
		std::vector<std_vector_Vector3d> * linaccs, // Out
		std::vector<std_vector_Vector3d> * angaccs, // Out
		std::vector<std_vector_Vector3d> * fs_in, // In
		std::vector<std_vector_Vector3d> * ts_in, // In
		std::vector<std_vector_Vector3d> * f_local, // Out
		std::vector<std_vector_Vector3d> * t_local, // Out
		std::vector<double> * com_mass, // Out
		std_vector_Vector3d * com_pos, // Out
		std_vector_Vector3d * com_linvel, // Out
		std_vector_Vector3d * com_ang_momentums, // Out
		std_vector_Matrix3d * com_inertia // Out
		)
	{
		size_t Size = q->size();
		dCASSERT(Size == dq->size());
		dCASSERT(Size == ddq->size());
		dCASSERT(Size == fs_in->size());
		dCASSERT(Size == ts_in->size());

		linvels->resize(Size);
		angvels->resize(Size);
		linaccs->resize(Size);
		angaccs->resize(Size);
		
		f_local->resize(Size);
		t_local->resize(Size);

		com_mass->resize(Size);
		com_pos->resize(Size);
		com_linvel->resize(Size);
		com_ang_momentums->resize(Size);
		com_inertia->resize(Size);
		for (size_t i = 0; i < Size; i++)
		{
			ptr->InitializeInverseDynamics((*q)[i], (*dq)[i], (*ddq)[i]);
			ptr->ComputeVelocityAcceleration(gravity->data);
			(*linvels)[i].data = ptr->GetLocalLinearVelocity();
			(*angvels)[i].data = ptr->GetLocalAngularVelocity();
			(*linaccs)[i].data = ptr->GetLocalLinearAcceleration();
			(*angaccs)[i].data = ptr->GetLocalAngularAcceleration();
			ptr->ComputeCoMMomentums((*com_mass)[i], com_pos->data[i], com_linvel->data[i], com_ang_momentums->data[i], com_inertia->data[i]);
			ptr->ComputeForceTorque((*fs_in)[i].data, (*ts_in)[i].data);
			
			(*f_local)[i].data = ptr->GetLocalForce();
			(*t_local)[i].data = ptr->GetLocalTorque();
		}
	}

	void InvDynComputeForceTorque(CInverseDynamicsPtr ptr, std_vector_Vector3d * force, std_vector_Vector3d * torque)
	{
		ptr->ComputeForceTorque(force->data, torque->data);
	}

	void InvDynGetLocalLinearVelocity(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr * out)
	{
		const std::vector<Vector3d> & res = ptr->GetLocalLinearVelocity();
		out->ptr = &res;
	}

	void InvDynGetLocalAngularVelocity(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
	{
		const std::vector<Vector3d>& res = ptr->GetLocalAngularVelocity();
		out->ptr = &res;
	}

	void InvDynGetLocalLinearAcceleration(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr * out)
	{
		const std::vector<Vector3d>& res = ptr->GetLocalLinearAcceleration();
		out->ptr = &res;
	}

	void InvDynGetLocalAngularAcceleration(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr * out)
	{
		const std::vector<Vector3d>& res = ptr->GetLocalAngularAcceleration();
		out->ptr = &res;
	}

	void InvDynGetLocalForce(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr * out)
	{
		const std::vector<Vector3d> & res = ptr->GetLocalForce();
		out->ptr = &res;
	}

	void InvDynGetLocalTorque(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
	{
		const std::vector<Vector3d>& res = ptr->GetLocalTorque();
		out->ptr = &res;
	}

	void InvDynGetJointRotation(CInverseDynamicsPtr ptr, std_vector_Matrix3d_ptr * out)
	{
		const std::vector<Matrix3d>& res = ptr->GetJointRotation();
		out->ptr = &res;
	}

	void InvDynGetBodyOrientation(const CInverseDynamicsPtr ptr, std_vector_Matrix3d_ptr* out)
	{
		const std::vector<Matrix3d> & res = ptr->GetBodyOrientation();
		out->ptr = &res;
	}

	int InvDygGetTotalJointDof(const CInverseDynamicsPtr ptr)
	{
		return static_cast<int>(ptr->GetTotalJointDof());
	}

	int InvDynGetJointDof(const CInverseDynamicsPtr ptr, int bid)
	{
		dCASSERT(ptr != NULL);
		return ptr->GetJointDof(bid);
	}

	void InvDynReRoot(CInverseDynamicsPtr ptr, int bodyId)
	{
		dCASSERT(ptr != NULL);
		ptr->ReRoot(bodyId);
	}

	void InvDynReComputeR(CInverseDynamicsPtr ptr)
	{
		dCASSERT(ptr != NULL);
		ptr->ReComputeR();
	}

	void InvDynClearReRootFlag(CInverseDynamicsPtr ptr)
	{
		dCASSERT(ptr != NULL);
		ptr->ClearReRootFlag();
	}

	void InvDynComputeBodyPositions(CInverseDynamicsPtr ptr, std_vector_Vector3d * out)
	{
		ptr->ComputeBodyPositions(out->data);
	}

	void InvDynComputeComMomentums(CInverseDynamicsPtr ptr, double * Mass, Eigen_Vector3d * com, 
		Eigen_Vector3d * linVelocity, Eigen_Vector3d * angMomentum, Eigen_Matrix3d * inertia)
	{
		dCASSERT(ptr != NULL);
		ptr->ComputeCoMMomentums(*Mass, com->data, linVelocity->data, angMomentum->data, inertia->data);
	}

	void CInvDynComputeComMomentums(CInverseDynamicsPtr ptr, double * Mass, Vector3d com,
		Vector3d linVelocity, Vector3d angMomentum,
		Matrix3d inertia)
	{
		dCASSERT(ptr != NULL);
		ptr->ComputeCoMMomentums(*Mass, com, linVelocity, angMomentum, inertia);
	}

	void InvDynComputeLocalW_q(const CInverseDynamicsPtr ptr, int bodyId, std_vector_Vector3d * out)
	{
		dCASSERT(ptr != NULL);
		ptr->ComputeLocalW_q(out->data, bodyId);
	}

	void InvDynGetJwJoint(CInverseDynamicsPtr ptr)
	{
		dCASSERT(ptr != NULL);
		// TODO
	}

	void InvDynGetJwDotJoint(CInverseDynamicsPtr ptr)
	{
		dCASSERT(ptr != NULL);
	}

	void InvDynGetCls(CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr * out)
	{
		const std::vector<Vector3d>& res = ptr->GetCls();
		out->ptr = &res;
	}

	void InvDynGetDCls(CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr * out)
	{
		dCASSERT(ptr != NULL);
		const std::vector<Vector3d>& res = ptr->GetDCls();
		out->ptr = &res;
	}

	void DestroyInverseDynamics(CInverseDynamicsPtr ptr)
	{
		if (ptr != NULL)
		{
			delete ptr;
		}
	}

	void DestroyInverseDynamicsHandle(size_t ptr)
	{
		DestroyInverseDynamics((CInverseDynamicsPtr)ptr);
	}

//#ifdef __cplusplus
//} // end extern "C"
//#endif