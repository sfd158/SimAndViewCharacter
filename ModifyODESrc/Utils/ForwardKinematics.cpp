
// input: root position, and joint local rotation
// output: global body position, orientation, 
# include "QuaternionWithGrad.h"
# include "ForwardKinematics.h"
# include <cstring>

void fk_joint_info(
	const double * root_position, // 
	const double * joint_rotation,
	const int * parent,
	const double * local_offset,
	double * joint_position,
	double * joint_orientation,
	int num_joints // root is included.
)
{
	for (int j = 0; j < 3; j++) joint_position[j] = root_position[j];
	for (int j = 0; j < 4; j++) joint_orientation[j] = joint_rotation[j];
	for (int i = 1; i < num_joints; i++)
	{
		int pa_id = parent[i];
		const double * pa_quat = joint_orientation + 4 * pa_id;
		double offset[3] = {0.0, 0.0, 0.0};
		quat_apply_single(pa_quat, local_offset + 3 * i, offset);
		vec3_add_vec3(joint_position + 3 * pa_id, offset, joint_position + 3 * i);
		double tmp_quat[4] = {0.0, 0.0, 0.0, 1.0};
		quat_multiply_single(pa_quat, joint_rotation + 4 * i, tmp_quat);
		vector_normalize_single(tmp_quat, 4, joint_orientation + 4 * i);
	}
}

void fk_joint_info_only_pos(
	const double * root_position, // 
	const double * joint_orientation,
	const int * parent,
	const double * local_offset,
	double * joint_position,
	int num_joints // root is included.
)
{
	for (int j = 0; j < 3; j++) joint_position[j] = root_position[j];
	for (int i = 1; i < num_joints; i++)
	{
		int pa_id = parent[i];
		const double * pa_quat = joint_orientation + 4 * pa_id;
		double offset[3] = {0.0, 0.0, 0.0};
		quat_apply_single(pa_quat, local_offset + 3 * i, offset);
		vec3_add_vec3(joint_position + 3 * pa_id, offset, joint_position + 3 * i);
	}
}

#define USE_SHARE_MEMORY_BUF 1

#if USE_SHARE_MEMORY_BUF
static double static_body_pos_buf[2 * 4 * 30];
static double static_body_quat_buf[2 * 4 * 30];
static double static_joint_pos_buf[4 * 30];
static double static_joint_quat_buf[4 * 30];
#endif

ReferenceMotionHandle::ReferenceMotionHandle(
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
):
	original_root_pos(root_pos_),
	original_joint_rotation(joint_rot_),
	original_joint_orient(joint_orient_),
	done(done_),
	original_h(original_h_),
	target_h(target_h_),
	num_frames(num_frames_),
	num_joints(num_joints_),
	dt(dt_),
	inv_dt(1.0 / dt),
	root_scale(target_h_ / original_h_)
{
	joint_parent = new int[num_joints];
	memcpy(joint_parent, joint_parent_, sizeof(int) * num_joints);
		
	child_body_index = new int[num_joints_];
	memcpy(child_body_index, child_body_index_, sizeof(int) * num_joints_);

	joint_offset = new double[4 * num_joints_];
	memcpy(joint_offset, joint_offset_, 4 * num_joints_ * sizeof(double));

	body_offset = new double[4 * num_joints_];
	memcpy(body_offset, body_offset_, 4 * num_joints_ * sizeof(double));

	#if USE_SHARE_MEMORY_BUF
	body_pos_buf = static_body_pos_buf;
	body_quat_buf = static_body_quat_buf;
	joint_pos_buf = static_joint_pos_buf;
	joint_quat_buf = static_joint_quat_buf;
	#else
	body_pos_buf = new double[2 * 4 * num_joints_];
	body_quat_buf = new double[2 * 4 * num_joints_];
	joint_pos_buf = new double[4 * num_joints];
	joint_quat_buf = new double[4 * num_joints];
	#endif
}

ReferenceMotionHandle::~ReferenceMotionHandle()
{
	if (joint_parent != nullptr) delete[] joint_parent;
	if (child_body_index != nullptr) delete[] child_body_index;
	if (joint_offset != nullptr) delete[] joint_offset;
	if (body_offset != nullptr) delete[] body_offset;

	#if USE_SHARE_MEMORY_BUF
	#else
	if (body_pos_buf != nullptr) delete[] body_pos_buf;
	if (body_quat_buf != nullptr) delete[] body_quat_buf;
	if (joint_pos_buf != nullptr) delete[] joint_pos_buf;
	if (joint_quat_buf != nullptr) delete[] joint_quat_buf;
	#endif
}

void ReferenceMotionHandle::get_body_pos_quat(int frame, double * ret_pos, double * ret_quat)
{
	double _root_pos[3] = {0.0, 0.0, 0.0};  // scale the root
	scale_vec3(root_scale, original_root_pos + 3 * frame, _root_pos);
	fk_joint_info(_root_pos, original_joint_rotation + 4 * frame * num_joints, joint_parent,
		joint_offset, joint_pos_buf, joint_quat_buf, num_joints);
	for (int i = 0; i < num_joints; i++)
	{
		int child = child_body_index[i];
		double delta[3] = {0, 0, 0};
		quat_apply_single(joint_quat_buf + 4 * i, body_offset + 3 * i, delta);
		for (int j = 0; j < 3; j++) ret_pos[3 * child + j] = joint_pos_buf[3 * i + j] + delta[j];
		for (int j = 0; j < 4; j++) ret_quat[4 * child + j] = joint_quat_buf[4 * i + j];
	}
}

void ReferenceMotionHandle::get_body_pos(int frame, double * ret_pos)
{
	double _root_pos[3] = {0.0, 0.0, 0.0};  // scale the root
	scale_vec3(root_scale, original_root_pos + 3 * frame, _root_pos);
	const double * joint_orient = original_joint_orient + 4 * frame * num_joints;
	fk_joint_info_only_pos(_root_pos, joint_orient, joint_parent, joint_offset, joint_pos_buf, num_joints);
	for (int i = 0; i < num_joints; i++)
	{
		int child = child_body_index[i];
		double delta[3] = {0, 0, 0};
		quat_apply_single(joint_orient + 4 * i, body_offset + 3 * i, delta);
		for (int j = 0; j < 3; j++) ret_pos[3 * child + j] = joint_pos_buf[3 * i + j] + delta[j];
	}
}

/*
* The joint local rotation is provided.
*/
void ReferenceMotionHandle::get_state(int frame, double * result)
{
	if (frame >= num_frames)
	{
		std::cout << "Frame = " << frame << " >= " << num_frames << std::endl;
		std::exit(1);
	}
	double * curr_pos_ptr = body_pos_buf;
	double * curr_quat_ptr = body_quat_buf;
	if (frame == 0 || done[frame - 1] == 1) // return the next frame..
	{
		frame += 1;
		curr_pos_ptr = body_pos_buf + 3 * num_joints;
		curr_quat_ptr = body_quat_buf + 4 * num_joints;
	} 

	get_body_pos_quat(frame, body_pos_buf, body_quat_buf); // compute the linear and angular velocity.
	get_body_pos_quat(frame - 1, body_pos_buf + 3 * num_joints, body_quat_buf + 4 * num_joints);
	
	for (int i = 0; i < num_joints; i++)
	{
		for (int j = 0; j < 3; j++) result[i * 13 + j] = curr_pos_ptr[i * 3 + j];
		for (int j = 0; j < 4; j++) result[i * 13 + 3 + j] = curr_quat_ptr[i * 4 + j];
		for (int j = 0; j < 3; j++) result[i * 13 + 7 + j] = inv_dt * (
			body_pos_buf[i * 3 + j] - body_pos_buf[3 * num_joints + i * 3 + j]);
		delta_quat2_single(body_quat_buf + 4 * num_joints + 4 * i, body_quat_buf + 4 * i, inv_dt, result + i * 13 + 10);
	}
}

/*
* The global joint orientation is provided.
*/
void ReferenceMotionHandle::get_state_with_orient(int frame, double * result)
{
	if (frame >= num_frames)
	{
		std::cout << "Frame = " << frame << " >= " << num_frames << std::endl;
		std::exit(1);
	}
	double * curr_pos_ptr = body_pos_buf;
	const double * curr_quat_ptr = original_joint_orient + 4 * frame * num_joints;
	const double * prev_quat_ptr = nullptr;
	bool is_first = frame == 0 || done[frame - 1] == 1;
	if (is_first) // return the next frame..
	{
		frame += 1;
		curr_pos_ptr = body_pos_buf + 3 * num_joints;
	} 

	get_body_pos(frame, body_pos_buf); // compute the linear and angular velocity.
	get_body_pos(frame - 1, body_pos_buf + 3 * num_joints);
	
	for (int i = 0; i < num_joints; i++)
	{
		for (int j = 0; j < 3; j++) result[i * 13 + j] = curr_pos_ptr[i * 3 + j];
		for (int j = 0; j < 4; j++) result[i * 13 + 3 + j] = curr_quat_ptr[i * 4 + j];
		for (int j = 0; j < 3; j++) result[i * 13 + 7 + j] = inv_dt * (
			body_pos_buf[i * 3 + j] - body_pos_buf[3 * num_joints + i * 3 + j]);
	}

	if (is_first) {prev_quat_ptr = curr_quat_ptr; curr_quat_ptr += 4 * num_joints;}
	else {prev_quat_ptr = curr_quat_ptr - 4 * num_joints;}
	for (int i = 0; i < num_joints; i++)
	{
		delta_quat2_single(prev_quat_ptr + 4 * i, curr_quat_ptr + 4 * i, inv_dt, result + i * 13 + 10);
	}
}

int ReferenceMotionHandle::get_num_joints() const
{
	return num_joints;
}

int ReferenceMotionHandle::get_num_frames() const
{
	return num_frames;
}

double ReferenceMotionHandle::get_dt() const
{
	return dt;
}

double ReferenceMotionHandle::get_root_scale() const
{
	return root_scale;
}

const int * ReferenceMotionHandle::joint_parent_ptr() const
{
	return joint_parent;
}

const int * ReferenceMotionHandle::child_body_index_ptr() const
{
	return child_body_index;
}

void ReferenceMotionHandle::print_root_pos(int frame) const
{
	const double * ptr = original_root_pos + 3 * frame;
	std::cout << "root_pos[" << frame << "] = " << ptr[0] << " " << ptr[1] << " " << ptr[2] << std::endl;
}

void ReferenceMotionHandle::print_local_rotation(int frame) const
{
	const double * ptr = original_joint_rotation + 4 * num_joints * frame;
	std::cout << "local_rotation" << std::endl;
	for(int i = 0; i < num_joints; i++)
	{
		const double * p = ptr + 4 * i;
		for(int j = 0; j < 4; j++)
		{
			std::cout << p[j] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl << std::endl;
}