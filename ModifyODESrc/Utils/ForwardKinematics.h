#pragma once
#include <iostream>

class ReferenceMotionHandle {
	// I know this is ugly, but this implementation saves memory.
	// save the data in numpy.
public:
	const double* original_root_pos = nullptr;
	const double* original_joint_rotation = nullptr;
	const double* original_joint_orient = nullptr;

	const int* done = nullptr;
	double original_h = 0;
	double target_h = 0;
	double root_scale = 0;

	int* joint_parent = nullptr;
	int* child_body_index = nullptr;
	double* joint_offset = nullptr;
	double* body_offset = nullptr;

	int num_frames = 0;
	int num_joints = 0;
	double dt = 0.0;
	double inv_dt = 0.0;

	double* body_pos_buf = nullptr;
	double* body_quat_buf = nullptr;

	double* joint_pos_buf = nullptr;
	double* joint_quat_buf = nullptr;

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
	);

	~ReferenceMotionHandle();
	void get_body_pos_quat(int frame, double* ret_pos, double* ret_quat);
	void get_body_pos(int frame, double * ret_pos);

	void get_state(int frame, double* result);
	void get_state_with_orient(int frame, double * result);
	int get_num_joints() const;
	int get_num_frames() const;
	double get_dt() const;
	double get_root_scale() const;
	const int * joint_parent_ptr() const;
	const int * child_body_index_ptr() const;

	void print_root_pos(int frame) const;
	void print_local_rotation(int frame) const;
};