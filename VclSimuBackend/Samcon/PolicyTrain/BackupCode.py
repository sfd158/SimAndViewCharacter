"""
if compute_facing and False:
    # convert to facing coordinate..
    facing_body_pos_list: np.ndarray = body_pos_list.copy()
    facing_body_pos_list[:, :, [0, 2]] -= body_pos_list[:, root_body_index: root_body_index + 1, [0, 2]]
    facing_body_velo_list: np.ndarray = body_vel_list.copy()
    facing_body_quat_list: np.ndarray = body_quat_list.copy()
    facing_body_omega_list: np.ndarray = body_omega_list.copy()

    root_quat_list: np.ndarray = np.ascontiguousarray(body_quat_list[:, root_body_index, :])
    facing_y, facing_xz = MathHelper.y_decompose(root_quat_list)
    facing_y_inv: Rotation = Rotation(facing_y)
    for body_index in range(num_body):
        # TODO: how to compute linear velocity in facing coordinate..?
        facing_body_pos_list[:, body_index, :] = facing_y_inv.apply(facing_body_pos_list[:, body_index, :])
        facing_body_quat_list[:, body_index, :] = (facing_y_inv * Rotation(facing_body_quat_list[:, body_index, :])).as_quat()
        facing_body_omega_list[:, body_index, :] = facing_y_inv.apply(facing_body_omega_list[:, body_index, :])

    facing_body_pos_list: np.ndarray = np.ascontiguousarray(facing_body_pos_list)
    facing_body_velo_list: np.ndarray = np.ascontiguousarray(facing_body_velo_list)
    facing_body_quat_list: np.ndarray = np.ascontiguousarray(facing_body_quat_list)
    facing_body_omega_list: np.ndarray = np.ascontiguousarray(facing_body_omega_list)
"""
