# cython: language_level=3
from EigenWrapper cimport *
from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string


cdef extern from "inversedynamicswrapper.h" nogil:
    ctypedef char CStr4[4]
    ctypedef double CVec3[3]
    ctypedef CVec3 CMat33[3]

    cdef cppclass CInverseDynamics:
        const std_vector[int] & GetJointQIndex() const
        size_t size() const
        void print_joint_axies() const

    ctypedef CInverseDynamics * CInverseDynamicsPtr

    CInverseDynamicsPtr InvDynCreate(
        const std_vector[double]* body_mass,
        const std_vector_Matrix3d* body_inertia,
        const std_vector_Vector3d* body_position,
        const std_vector_Matrix3d* body_rotation,
        const std_vector[int]* parent_joint_dof,
        const std_vector_Vector3d* parent_joint_pos,
        const std_vector[std_string]* parent_joint_euler_order,
        const std_vector_Matrix3d* parent_joint_euler_axis,
        const std_vector[int]* parent_body_index)

    int InvDynGeneralizedCoordinatesDimension(CInverseDynamicsPtr ptr)
    void InvDynConvertToGeneralizeCoordinates(
        CInverseDynamicsPtr ptr,
        const Eigen_Vector3d* rootPos, # In
        const std_vector_Quaterniond* vJointRots, # In
        std_vector[double]* q # Out
    )

    void InvDynConvertToGeneralizeCoordinatesBatch(
        CInverseDynamicsPtr ptr,
        const std_vector_Vector3d * rootPosBatch,
        const std_vector_std_vector_Quaterniond * vJointRotsBatch,
        std_vector[std_vector[double]] * q
    )

    void InvDynConvertToJointRotations(
        CInverseDynamicsPtr ptr,
        const std_vector[double]* q,
        Eigen_Vector3d* rootPos,
        std_vector_Quaterniond* vJointRots
    )

    void InvDynInitializeInverseDynamics(
        CInverseDynamicsPtr ptr,
        const std_vector[double]* q,
        const std_vector[double]* qdot,
        const std_vector[double]* qdotdot
	)

    void InvDynComputeVelocityAcceleration(CInverseDynamicsPtr ptr, double gravity_x, double gravity_y, double gravity_z)
    void InvDynComputeForceTorque(CInverseDynamicsPtr ptr, std_vector_Vector3d * force, std_vector_Vector3d * torque)

    void InvDynComputeForceTorqueBatch(
        CInverseDynamicsPtr ptr,
        const std_vector[std_vector[double]]* q,
        const std_vector[std_vector[double]]* dq,
        const std_vector[std_vector[double]]* ddq,
        const Eigen_Vector3d* gravity,
        std_vector[std_vector_Vector3d] * linvels, # Out
        std_vector[std_vector_Vector3d] * angvels, # Out
        std_vector[std_vector_Vector3d] * linaccs, # Out
        std_vector[std_vector_Vector3d] * angaccs, # Out
        std_vector[std_vector_Vector3d] * fs_in, # In
        std_vector[std_vector_Vector3d] * ts_in, # In
        std_vector[std_vector_Vector3d] * f_local, # Out
        std_vector[std_vector_Vector3d] * t_local, # Out
        std_vector[double]* com_mass, # Out
        std_vector_Vector3d* com_pos, # Out
        std_vector_Vector3d* com_linvel, # Out
        std_vector_Vector3d* com_ang_momentums, # Out
        std_vector_Matrix3d* com_inertia # Out
	)

    void InvDynGetLocalLinearVelocity(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
    void InvDynGetLocalAngularVelocity(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
    void InvDynGetLocalLinearAcceleration(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
    void InvDynGetLocalAngularAcceleration(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)

    void InvDynGetLocalForce(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
    void InvDynGetLocalTorque(const CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)

    void InvDynGetJointRotation(CInverseDynamicsPtr ptr, std_vector_Matrix3d_ptr* out)
    void InvDynGetBodyOrientation(const CInverseDynamicsPtr ptr, std_vector_Matrix3d_ptr* out)
	
    int InvDygGetTotalJointDof(const CInverseDynamicsPtr ptr)
    int InvDynGetJointDof(const CInverseDynamicsPtr ptr, int bid)
    void InvDynReRoot(CInverseDynamicsPtr ptr, int bodyId)
    void InvDynReComputeR(CInverseDynamicsPtr ptr)
    void InvDynClearReRootFlag(CInverseDynamicsPtr ptr)
    void InvDynComputeBodyPositions(CInverseDynamicsPtr ptr, std_vector_Vector3d* out)
    void InvDynComputeComMomentums(CInverseDynamicsPtr ptr, double* Mass, 
                                   Eigen_Vector3d* com, Eigen_Vector3d* linVelocity,
                                   Eigen_Vector3d* angMomentum, 
                                   Eigen_Matrix3d* inertia)
    void InvDynComputeLocalW_q(const CInverseDynamicsPtr ptr, int bodyId, std_vector_Vector3d* out)
    void InvDynGetCls(CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
    void InvDynGetDCls(CInverseDynamicsPtr ptr, std_vector_Vector3d_ptr* out)
    void DestroyInverseDynamics(CInverseDynamicsPtr)
    void DestroyInverseDynamicsHandle(size_t)


cdef extern from "PDControlAdd.h" nogil:
    cdef cppclass TorqueAddHelper:
        TorqueAddHelper()
        TorqueAddHelper(std_vector[int]& parent_body_, std_vector[int]& child_body_, int body_cnt_)

        int GetBodyCount() const
        int GetJointCount() const
        const std_vector[int] * GetParentBody() const
        const std_vector[int] * GetChildBody() const

        const int * get_parent_body_c_ptr() const
        const int * get_child_body_c_ptr() const
        const double * get_kp_c_ptr() const
        const double * get_max_len_c_ptr() const
        void set_pd_control_param(const std_vector[double] & kp_, const std_vector[double] & max_len_)

        void backward(const double * prev_grad, double * out_grad)
        void add_torque_forward(double* body_torque, const double * joint_torque)

    TorqueAddHelper* TorqueAddHelperCreate(const std_vector[int]& parent_body_,
        const std_vector[int]& child_body_, size_t body_cnt_)
    void TorqueAddHelperDelete(TorqueAddHelper* ptr)


cdef extern from "MixQuaternion.h" nogil:
    void mix_quaternion(double * quat_input, size_t num, double * result) # The performance of this function is not good..


cdef extern from "QuaternionWithGrad.h" nogil:
    void quat_multiply_single(
        const double * q1,
        const double * q2,
        double * q
    )

    void quat_multiply_single_float32(
        const float * q1,
        const float * q2,
        float * q
    )

    void quat_inv_impl(
        const double * q,
        double * out_q,
        size_t num_quat
    )

    void quat_inv_single(
        const double * q,
        double * out_q
    )

    void quat_inv_backward_single(
        const double * q,
        const double * grad_in,
        double * grad_out
    )

    void quat_inv_backward_impl(
        const double * q,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_multiply_forward(
        const double * q1,
        const double * q2,
        double * q,
        size_t num_quat
    )

    void quat_multiply_forward_float32(
        const float * q1,
        const float * q2,
        float * q,
        size_t num_quat
    )

    void quat_multiply_backward_single(
        const double * q1,
        const double * q2,
        const double * grad_q, # \frac{\partial L}{\partial q_x, q_y, q_z, q_w}
        double * grad_q1,
        double * grad_q2
    )

    void quat_multiply_backward(
        const double * q1,
        const double * q2,
        const double * grad_q,
        double * grad_q1,
        double * grad_q2,
        size_t num_quat
    )

    void quat_apply_single(
        const double * q,
        const double * v,
        double * o
    )

    void quat_apply_forward(
        const double * q,
        const double * v,
        double * o,
        size_t num_quat
    )

    void quat_apply_backward_single(
        const double * q,
        const double * v,
        const double * o_grad,
        double * q_grad,
        double * v_grad
    )

    # Add by Yulong Zhang
    void quat_apply_forward_one2many(
        const double * q,
        const double * v,
        double * o,
        size_t num_quat
    )

    void quat_apply_backward_single(
        const double * q,
        const double * v,
        const double * o_grad,
        double * q_grad,
        double * v_grad
    )

    void quat_apply_backward(
        const double * q,
        const double * v,
        const double * o_grad,
        double * q_grad,
        double * v_grad,
        size_t num_quat
    )

    void flip_quat_by_w_forward_impl(
        const double * q,
        double * q_out,
        size_t num_quat
    )

    void flip_quat_by_w_backward_impl(
        const double * q,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_to_vec6d_single(
        const double * q,
        double * vec6d
    )

    void quat_to_vec6d_impl(const double * q, double * vec6d, size_t num_quat)

    void quat_to_matrix_forward_single(
        const double * q,
        double * mat
    )

    void quat_to_matrix_impl(
        const double * q,
        double * mat,
        size_t num_quat
    )

    void quat_to_matrix_backward_single(
        const double * q,
        const double * grad_in,
        double * grad_out
    )

    void quat_to_matrix_backward(
        const double * q,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )
    # Add by Yulong Zhang
    void six_dim_mat_to_quat_single(
        const double * mat,
        double * quat
    )
    void six_dim_mat_to_quat_impl(
        const double * mat,
        double * q,
        size_t num_quat
    )
    void vector_to_cross_matrix_single(
        const double * vec,
        double * mat
    )

    void vector_to_cross_matrix_impl(
        const double * vec,
        double * mat,
        size_t num_vec
    )

    void vector_to_cross_matrix_backward_single(
        const double * vec,
        const double * grad_in,
        double * grad_out
    )

    void vector_to_cross_matrix_backward(
        const double * vec,
        const double * grad_in,
        double * grad_out,
        size_t num_vec
    )

    void quat_to_rotvec_single(
        const double * q,
        double & angle,
        double * rotvec
    )

    void quat_to_rotvec_impl(
        const double * q,
        double * angle,
        double * rotvec,
        size_t num_quat
    )

    void quat_to_rotvec_backward_single(
        const double * q,
        double angle,
        const double * grad_in,
        double * grad_out
    )

    void quat_to_rotvec_backward(
        const double * q,
        const double * angle,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_from_rotvec_single(
        const double * rotvec,
        double * q
    )

    void quat_from_rotvec_impl(
        const double * rotvec,
        double * q,
        size_t num_quat
    )

    void quat_from_rotvec_backward_single(
        const double * rotvec,
        const double * grad_in,
        double * grad_out
    )

    void quat_from_rotvec_backward_impl(
        const double * rotvec,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_from_matrix_single(
        const double * mat,
        double * q
    )

    void quat_from_matrix_impl(
        const double * mat,
        double * q,
        size_t num_quat
    )

    void quat_from_matrix_backward_single(
        const double * mat,
        const double * grad_in,
        double * grad_out
    )

    void quat_from_matrix_backward_impl(
        const double * mat,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_to_hinge_angle_single(
        const double * q,
        const double * axis,
        double & angle
    )

    void quat_to_hinge_angle_forward(
        const double * q,
        const double * axis,
        double * angle,
        size_t num_quat
    )

    void quat_to_hinge_angle_backward_single(
        const double * q,
        const double * axis,
        double grad_in,
        double * grad_out
    )

    void quat_to_hinge_angle_backward(
        const double * q,
        const double * axis,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void parent_child_quat_to_hinge_angle(
        const double * quat0,
        const double * quat1,
        const double * init_rel_quat_inv,
        const double * axis,
        double * angle,
        size_t num_quat
    )

    void parent_child_quat_to_hinge_angle_backward(
        const double * quat0,
        const double * quat1,
        const double * init_rel_quat_inv,
        const double * axis,
        const double * grad_in,
        double * quat0_grad,
        double * quat1_grad,
        size_t num_quat
    )

    void quat_integrate_impl(
        const double * q,
        const double * omega,
        double dt,
        double * result,
        size_t num_quat
    )

    void quat_integrate_backward(
        const double * q,
        const double * omega,
        double dt,
        const double * grad_in,
        double * q_grad,
        double * omega_grad,
        size_t num_quat
    )

    void vector_normalize_single(
        const double * x,
        size_t ndim,
        double * result
    )

    void vector_normalize_backward_single(
        const double * x,
        size_t ndim,
        const double * grad_in,
        double * grad_out
    )

    void normalize_quaternion_impl(
        const double * q_in,
        double * q_out,
        size_t num_quat
    )

    void quat_integrate_single(
        const double * q,
        const double * omega,
        double dt,
        double * result
    )

    void quat_integrate_impl(
        const double * q,
        const double * omega,
        double dt,
        double * result,
        size_t num_quat
    )

    void quat_integrate_backward_single(
        const double * q,
        const double * omega,
        double dt,
        const double * grad_in,
        double * q_grad,
        double * omega_grad
    )

    void quat_integrate_backward(
        const double * q,
        const double * omega,
        double dt,
        const double * grad_in,
        double * q_grad,
        double * omega_grad,
        size_t num_quat
    )

    void delta_quat2_impl_float32(
        const float* prev_q,
        const float* q,
        float inv_dt,
        float* result,
        size_t num_quat
    )

    void cross_product_single(const double * a, const double * b, double * c)
    void quat_between_single(const double* a, const double* b, double* result)

    # Add by Yulong Zhang
    void calc_surface_distance_to_capsule(
        const double * relative_pos,
        size_t ndim,
        double radius,
        double length,
        double * sd,
        double * normal
    )

    void clip_vec_by_norm_forward_single(
        const double * x,
        double min_val,
        double max_val,
        double * result,
        size_t ndim
    )

    void clip_vec_by_norm_backward_single(
        const double * x,
        double min_val,
        double max_val,
        const double * grad_in,
        double * grad_out,
        size_t ndim
    )

    void clip_vec_by_length_forward(
        const double * x,
        double max_len,
        double * result,
        size_t ndim
    )

    void clip_vec3_arr_by_length_forward(
        const double * x,
        const double * max_len,
        double * result,
        size_t num_vecs
    )

    void clip_vec_by_length_backward(
        const double * x,
        double max_len,
        const double * grad_in,
        double * grad_out,
        size_t ndim
    )

    void clip_vec3_arr_by_length_backward(
        const double * x,
        const double * max_len,
        const double * grad_in,
        double * grad_out,
        size_t num_vecs
    )

    void decompose_rotation_single(
        const double * q,
        const double * vb,
        double * result
    )

    void decompose_rotation(
        const double * q,
        const double * v,
        double * result,
        size_t num_quat
    )

    void decompose_rotation_pair_single(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b
    )

    void decompose_rotation_pair(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b,
        size_t num_quat
    )

    void decompose_rotation_pair_one2many(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b,
        size_t num_quat
    )
    void decompose_rotation_backward_single(
        const double * q,
        const double * v,
        const double * grad_in,
        double * grad_q,
        double * grad_v
    )

    void decompose_rotation_backward(
        const double * q,
        const double * v,
        const double * grad_in,
        double * grad_q,
        double * grad_v,
        size_t num_quat
    )


cdef extern from "ForwardKinematics.h" nogil:
    cdef cppclass ReferenceMotionHandle:
        const double* original_root_pos
        const double* original_joint_rotation
        const int* done
        double original_h
        double target_h
        double root_scale

        int* joint_parent
        int* child_body_index
        double* joint_offset
        double* body_offset

        int num_frames
        int num_joints
        double dt
        double inv_dt

        double* body_pos_buf
        double* body_quat_buf

        double* joint_pos_buf
        double* joint_quat_buf

        ReferenceMotionHandle(
            const double * root_pos_,
            const double * joint_rot_,
            const double * joint_orient_,
            const int * done_,
            double original_h_,
            double target_h_,
            const int * joint_parent_,
            const int * child_body_index_,
            const double * joint_offset_,
            const double * body_offset_,
            int num_frames_,
            int num_joints_,
            double dt_
        )

        void get_body_pos_quat(int frame, double* ret_pos, double* ret_quat)
        void get_body_pos(int frame, double * ret_pos)
        void get_state(int frame, double* result)
        void get_state_with_orient(int frame, double * result);
        int get_num_joints() const
        int get_num_frames() const
        double get_dt() const
        double get_root_scale() const
        const int * joint_parent_ptr() const
        const int * child_body_index_ptr() const
