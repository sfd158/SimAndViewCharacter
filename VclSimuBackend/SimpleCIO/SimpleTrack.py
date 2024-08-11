"""

The idea is come from paper:
Physics-based Human Motion Estimation and Synthesis from Videos

Optimize Variable
1. input 3d joint local rotation + root position
(We can only optimize root and 2 legs.
The upper part can be ignored..)
2. contact force of each time

Loss:
1. The optimized trajectory should be close to input motion
2. contact force + gravity should be close to
    linear acc of CoM and angular acc of inertia
3. loss of friction cone
4. etc

Impl details:
1. view toe and heel as end effector
2. The contact normal force is always along y axis
3. friction force is in character facing coordinate
4. About end effector: first load character as T-Pose
Then compute end effector position using box position and shape (as foot shape is box)

Optimize pipeline:
use inverse dynamcis result as initial solution
for each epoch:
    1. forward kinematics to compute global position,
    2. Then compute CoM and angular momentum
    3. compute contact label, (larger contact force has larger contact ratio)
    4. when the contact ratio is large, the distance from end effector to floor should be small..
    5. do back-prop, and optimize


TODO:
1. write all of hyper-parameters to a json config file
2. Try to track a single walking motioin
3. use inverse dynamics algorithm in Liu. et. al Sig Asia 2013 for initial solution
4. Add line search for optimization
5. enable / disable contact label as optimize variable
6. use spline for optimization?

Some tricks:
1. When computing linear/angular acc, we should consider boundary (the start, the end..)
"""

from enum import IntEnum
import ModifyODE as ode
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import torch
from torch import nn
from torch.nn import functional
from typing import List, Optional, Tuple, Union, Dict, Any

from ..Common.MathHelper import RotateType
from ..pymotionlib.MotionData import MotionData
from ..pymotionlib.PyTorchMotionData import PyTorchMotionData
from ..pymotionlib import BVHLoader
from ..DiffODE import DiffQuat
from ..DiffODE.PyTorchMathHelper import PyTorchMathHelper
from ..ODESim.BVHToTarget import BVHToTargetBase
from ..ODESim.TargetPose import SetTargetToCharacter
from ..ODESim.ODEScene import ODEScene, SceneContactInfo
from ..ODESim.ODECharacter import ODECharacter
from ..Server.v2.ServerForUnityV2 import ServerThreadHandle

from ..Utils import TorchMotionProjection


class OptimizeVarType(IntEnum):
    ALL_BODY = 0
    ROOT_AND_LEG = 1
    ONLY_KINEMATIC = 2


class WeightSummary:
    """
    Loss weights (or hyper parameters) of CIO algorithm for tracking 3d human pose
    """
    def __init__(self):
        self.w_friction: float = 100.0
        self.w_linear_momentum: float = 50.0
        self.w_angular_momentum: float = 60.0
        self.w_height_avoid: float = 50.0
        self.w_contact_penality_pos: float = 200.0
        self.w_contact_penality_vel: float = 10.0

        self.w_smooth_linear_acc: float = 1e-1
        self.w_smooth_angular_acc: float = 1e-3
        self.w_reference_2d: float = 10.0

        self.contact_label_k1: float = 0.05
        self.contact_label_k2: float = 3

        self.normalize()

    def update(self, conf: Dict[str, float]):
        self.w_friction: float = conf.get("w_friction", self.w_friction)
        self.w_linear_momentum: float = conf.get("w_linear_momentum", self.w_linear_momentum)
        self.w_angular_momentum: float = conf.get("w_angular_momentum", self.w_angular_momentum)
        self.w_height_avoid: float = conf.get("w_height_avoid", self.w_height_avoid)
        self.w_contact_penality_pos: float = conf.get("w_contact_penality_pos", self.w_contact_penality_pos)
        self.w_contact_penality_vel: float = conf.get("w_contact_penality_vel", self.w_contact_penality_vel)
        self.w_smooth_linear_acc: float = conf.get("w_smooth_linear_acc", self.w_smooth_linear_acc)
        self.w_smooth_angular_acc: float = conf.get("w_smooth_angular_acc", self.w_smooth_angular_acc)
        self.w_reference_2d: float = conf.get("w_reference_2d", self.w_reference_2d)
        self.contact_label_k1: float = conf.get("contact_label_k1", self.contact_label_k1)
        self.contact_label_k2: float = conf.get("contact_label_k2", self.contact_label_k2)

        self.normalize()

    def normalize(self):
        sum_val = sum(self.loss_list())
        self.w_friction /= sum_val
        self.w_linear_momentum /= sum_val
        self.w_angular_momentum /= sum_val
        self.w_height_avoid /= sum_val
        self.w_contact_penality_pos /= sum_val
        self.w_contact_penality_vel /= sum_val

        self.w_smooth_linear_acc /= sum_val
        self.w_smooth_angular_acc /= sum_val
        self.w_reference_2d /= sum_val

    def loss_list(self):
        return [self.w_friction, self.w_linear_momentum, self.w_angular_momentum, self.w_height_avoid, self.w_contact_penality_pos,
                self.w_contact_penality_vel, self.w_smooth_linear_acc, self.w_smooth_angular_acc, self.w_reference_2d]

    def export_as_dict(self):
        return {}.update(self.__dict__)


class SimpleTrack:
    # for finding foot joint and body index
    l_heel_joint_name = "lAnkle"
    l_heel_body_name = "lFoot"
    r_heel_joint_name = "rAnkle"
    r_heel_body_name = "rFoot"

    l_toe_joint_name = "lToeJoint"
    l_toe_body_name = "lToes"
    r_toe_joint_name = "rToeJoint"
    r_toe_body_name = "rToes"

    def __init__(
            self,
            scene: ODEScene,
            character: ODECharacter,
            motion: Union[str, MotionData],
            optimize_var_type: OptimizeVarType = OptimizeVarType.ROOT_AND_LEG,
            rotate_type: RotateType = RotateType.AxisAngle,
            only_optimize_force: bool = False,
            device=torch.device("cpu")
    ) -> None:
        self.device = device

        # for visualize in Unity Client
        self.run_as_sub_thread: bool = False
        self.export_target_set: Optional[SetTargetToCharacter] = None
        self.export_mocap: Optional[MotionData] = None

        self.body_contact_count = 2
        self.epoch: int = 0
        self.max_epoch = 2333
        self.contact_mu: float = 1.0  # default contact fraction
        self.lr = 0.1
        self.rotate_type: RotateType = rotate_type
        self.loss_weight = WeightSummary()
        fdir = os.path.dirname(__file__)
        self.dump_dir = os.path.join(fdir, "log_dir")
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        self.dump_fname: str = os.path.join(self.dump_dir, "simple-cio.ckpt")

        self.scene: ODEScene = scene
        self.character: ODECharacter = character
        self.gravity: torch.Tensor = torch.as_tensor(self.scene.gravity_numpy.flatten())  # (3,)

        self.character_body_names: List[str] = character.body_info.get_name_list()  # len == num bodies
        self.character_joint_names: List[str] = character.joint_info.joint_names()  # len == num joints

        if isinstance(motion, str):
            motion: MotionData = BVHLoader.load(motion)
        motion = motion.remove_end_sites()
        self.motion: MotionData = motion
        self.diff_motion = PyTorchMotionData()
        self.diff_motion.build_from_motion_data(self.motion)
        self.num_frames = motion.num_frames

        # Middle variables
        self.clear(False)

        self.end_site_offset: Optional[torch.Tensor] = None  # (4, 2, 3)
        self.end_site_parent: Optional[List[int]] = None  # (4, 2)

        self.compute_foot_contact_end_site()  # extract foot contact info

        self.bvh_child_body_offset: Optional[torch.Tensor] = None
        self.bvh_child_body_mass: Optional[torch.Tensor] = None  # mass of all bodies
        self.bvh_child_body_inertia: Optional[torch.Tensor] = None  # initial inertia of all bodies
        self.total_mass: Optional[float] = None  # mass of all bodies
        self.prepare_compute_body_info()

        if optimize_var_type == OptimizeVarType.ALL_BODY:
            self.joint_optim_index: List[int] = list(range(motion.num_frames))  # len == num_frames
        elif optimize_var_type == OptimizeVarType.ROOT_AND_LEG:  # only optimize root / leg
            # for std-human model
            optimize_names: List[str] = [
                "RootJoint", "lHip", "lKnee", "lAnkle", "lToeJoint",
                "rHip", "rKnee", "rAnkle", "rToeJoint"
            ]
            self.joint_optim_index: List[int] = [motion.joint_names.index(name) for name in optimize_names]
            self.joint_optim_index.sort()
        else:
            raise ValueError

        self.knee_optim_index = [self.joint_optim_index.index(motion.joint_names.index(knee_name)) for knee_name in ["lKnee", "rKnee"]]
        # Parameter Type
        rotate_shape: Tuple = (self.num_frames, len(self.joint_optim_index))
        rotate_inputs = self.diff_motion.joint_rotation[:, self.joint_optim_index, :].clone()

        # ground truth of 3d pose result
        self.rotate_gt: torch.Tensor = self.diff_motion.joint_rotation[:, self.joint_optim_index, :].detach().clone()
        self.root_pos_gt: torch.Tensor = self.diff_motion.joint_translation[:, 0, :].detach().clone()

        # build optimize parameter
        if self.rotate_type == RotateType.AxisAngle:
            rotate_shape = rotate_shape + (3,)
            rotate_inputs = DiffQuat.quat_to_rotvec(rotate_inputs.view(-1, 4)).view(rotate_shape)
            # convert ground truth as axis angle format
            self.rotate_gt: torch.Tensor = DiffQuat.quat_to_rotvec(self.rotate_gt.view(-1, 4)).view(rotate_shape)
        elif self.rotate_type == RotateType.Vec6d:
            rotate_shape = rotate_shape + (3, 2)
            rotate_inputs = DiffQuat.quat_to_vec6d(rotate_inputs.view(-1, 4)).view(rotate_shape)
            # convert ground truth as 6d vector
            self.rotate_gt: torch.Tensor = DiffQuat.quat_to_vec6d(self.rotate_gt.view(-1, 4)).view(rotate_shape)
        else:
            raise NotImplementedError

        # TODO: Load from file
        self.rotate_inputs_parameter = nn.Parameter(rotate_inputs, requires_grad=not only_optimize_force)

        # root position parameter
        root_pos_input = self.diff_motion.joint_translation[:, 0, :].detach().clone()
        self.root_pos_parameter = nn.Parameter(root_pos_input, requires_grad=False)

        # TODO: Load from file, or pre compute using inverse dynamics algorithm
        # The initial contact force should not initialized by zero.
        # it should be initialized by initial linear momentum
        # contact_force = torch.zeros(contact_shape, dtype=torch.float64)
        with torch.no_grad():
            self.contact_force_parameter = torch.zeros(
                (self.num_frames, len(self.end_site_parent), self.body_contact_count, 3), dtype=torch.float64
            )
            # self.compute_required_data()
            # we can divide contact force with height of end effectors.
            # use inverse dynamics method in paper:
            # Simulation and Control of Skeleton-driven Soft Body Characters, Liu et al. SIGGRAPH Asia 2013
            # contact point with smaller height has larger contact force.
            # if height of contact > 10cm, there is no contact force on this point
            contact_force = self.initialize_contact_force(True, False)

            # here we can initialize the target setter..
            self.initialize_bvh_target_setter()
            self.clear()

        self.contact_force_parameter = nn.Parameter(contact_force, requires_grad=False)

        # Parameter
        self.load()
        with torch.no_grad():
            self.compute_required_data()
        optim_list = [self.contact_force_parameter]
        if not only_optimize_force:
            optim_list.extend([self.rotate_inputs_parameter, self.root_pos_parameter])

        self.optim = torch.optim.LBFGS(optim_list, lr=self.lr, max_iter=200) #, line_search_fn="strong_wolfe")
        # self.optim = torch.optim.SGD(optim_list, lr=self.lr)
        print("build optim")
        self.loss_dict = None

    def build_only_kinematic_optimize(self):
        self.contact_force_parameter.requires_grad_ = False
        self.rotate_inputs_parameter.requires_grad_ = True
        self.root_pos_parameter.requires_grad_ = True
        optim_list = [self.rotate_inputs_parameter, self.root_pos_parameter]
        self.optim = torch.optim.LBFGS(optim_list, lr=self.lr, max_iter=100)

    def build_only_force_optimize(self):
        self.contact_force_parameter.requires_grad_ = True
        optim_list = [self.contact_force_parameter]
        self.rotate_inputs_parameter.requires_grad_ = False
        self.root_pos_parameter.requires_grad_ = False
        self.optim = torch.optim.LBFGS(optim_list, lr=self.lr, max_iter=100)

    def build_total_optimize(self):
        optim_list = [self.rotate_inputs_parameter, self.root_pos_parameter, self.contact_force_parameter]
        self.rotate_inputs_parameter.requires_grad_ = True
        self.root_pos_parameter.requires_grad_ = True
        self.contact_force_parameter.requires_grad_ = True
        self.optim = torch.optim.LBFGS(optim_list, lr=self.lr, max_iter=100)

    def initialize_bvh_target_setter(self):
        """
        for visualize in Unity client
        """
        self.export_mocap: MotionData = self.diff_motion.export_to_motion_data()
        self.export_mocap.recompute_joint_global_info()
        target = BVHToTargetBase(self.export_mocap, self.export_mocap.fps, self.character).init_target()
        self.export_target_set: Optional[SetTargetToCharacter] = SetTargetToCharacter(self.character, target)

    def initialize_contact_force(self, pre_compute: bool = True, do_clear: bool = False) -> torch.Tensor:
        with torch.no_grad():
            if pre_compute:
                self.compute_required_data()
            global_contact_force: torch.Tensor = self.calc_require_force()  # (num frame, 3)
            # convert to facing coordinate
            quat_y_inv: torch.Tensor = DiffQuat.quat_inv(self.facing_quat_y)
            contact_force: torch.Tensor = DiffQuat.quat_apply(quat_y_inv, global_contact_force)
            contact_force[..., 1][contact_force[..., 1] < 0] = 0
            end_height: torch.Tensor = self.contact_global_pos[..., 1] - torch.min(self.contact_global_pos[..., 1]) + torch.as_tensor(1e-4)
            end_ratio: torch.Tensor = torch.as_tensor(1.0) / end_height
            end_ratio[end_height > 0.1] = 0.0

            sum_ratio: torch.Tensor = torch.sum(end_ratio.view(self.num_frames, -1), dim=-1).view(self.num_frames, 1, 1) + torch.as_tensor(1e-8)
            end_ratio /= sum_ratio
            contact_force_ret: torch.Tensor = contact_force.view(self.num_frames, 1, 1, 3) * end_ratio[..., None].repeat(1, 1, 1, 3)

            # clip contact force on x and z component..
            x_force = contact_force_ret[..., 0]
            y_force = contact_force_ret[..., 1]
            sel_x_idx = torch.abs(x_force) > self.contact_mu * y_force
            x_force[sel_x_idx] = self.contact_mu * y_force[sel_x_idx]

            z_force = contact_force_ret[..., 2]
            sel_z_ix = torch.abs(z_force) > self.contact_mu * y_force
            z_force[sel_z_ix] = self.contact_mu * y_force[sel_z_ix]

        if do_clear:
            self.clear()
        return contact_force_ret

    def clear(self, clear_motion: bool = True):
        """
        clear middle states..
        """
        self.facint_quat_y: Optional[torch.Tensor] = None  # (frame, 4)
        self.child_body_position: Optional[torch.Tensor] = None  # position for all child bodies (frame, joint, 3)
        self.child_body_velocity: Optional[torch.Tensor] = None  # velocity for all child bodies (frame, joint, 3)

        self.global_com: Optional[torch.Tensor] = None  # global center of mass (frame, 3)
        self.linear_momentum: Optional[torch.Tensor] = None  # (frame, 3)
        self.delta_linear_momentum: Optional[torch.Tensor] = None  # (frame, 3)

        self.joint_angular_velo: Optional[torch.Tensor] = None  # (frame, joint, 3)
        self.joint_angular_acc: Optional[torch.Tensor] = None  # (frame, joint, 3)
        self.joint_linear_acc: Optional[torch.Tensor] = None  # (frame, joint, 3)

        self.angular_momentum: Optional[torch.Tensor] = None  # (frame, 3)
        self.delta_angular_momentum: Optional[torch.Tensor] = None  # (frame, 3)

        self.contact_global_force: Optional[torch.Tensor] = None  # (frame, 4, 2, 3)
        self.contact_global_torque: Optional[torch.Tensor] = None  # (frame, 4, 2, 3)
        self.contact_global_pos: Optional[torch.Tensor] = None  # (frame, 4, 2, 3)
        self.contact_global_velo: Optional[torch.Tensor] = None  # (frame, 4, 2, 3)

        self.contact_label: Optional[torch.Tensor] = None  # (frame, 4, 2)

        if clear_motion:
            self.diff_motion.clear()

        self.scene.contact_info = None  # clear contact info of previous step
        self.scene.str_info = ""

    def prepare_compute_body_info(self):
        # get the corresponding body index
        self.character.load_init_state()
        ode_child_list = [0, ]
        ode_parent_list = [-1, ]
        child_body_offset = np.zeros((self.motion.num_joints, 3), dtype=np.float64)
        child_body_mass = np.zeros((self.motion.num_joints,), dtype=np.float64)
        child_body_inertia = np.zeros((self.motion.num_joints, 3, 3), dtype=np.float64)

        for bvh_joint_idx, joint_name in enumerate(self.motion.joint_names):
            if bvh_joint_idx > 0:
                ode_joint_index: int = self.character_joint_names.index(joint_name)
                ode_joint: ode.Joint = self.character.joints[ode_joint_index]
                # get parent and child body..
                child_body: ode.Body = ode_joint.body1
                parent_body: ode.Body = ode_joint.body2
                ode_child_index = self.character.bodies.index(child_body)
                ode_parent_index = self.character.bodies.index(parent_body)

                ode_child_list.append(ode_child_index)
                ode_parent_list.append(ode_parent_index)

                # get child body position
                child_body_pos: np.ndarray = child_body.PositionNumpy
                joint_anchor: np.ndarray = ode_joint.getAnchorNumpy()
                child_body_offset[bvh_joint_idx, :] = child_body_pos - joint_anchor
            else:
                child_body: ode.Body = self.character.body_info.root_body

            child_mass: ode.Mass = child_body.getMass()
            child_body_mass[bvh_joint_idx] = child_mass.mass
            child_body_inertia[bvh_joint_idx] = child_mass.inertia.reshape((3, 3))

        # (num body, 3)
        self.bvh_child_body_offset: torch.Tensor = torch.as_tensor(child_body_offset, dtype=torch.float64)

        # (num body, )
        self.bvh_child_body_mass: torch.Tensor = torch.as_tensor(child_body_mass, dtype=torch.float64)
        self.total_mass = torch.sum(self.bvh_child_body_mass)

        # (num body, 3, 3)
        self.bvh_child_body_inertia: torch.Tensor = torch.as_tensor(child_body_inertia, dtype=torch.float64)

    def compute_child_body_pos_fast(self) -> torch.Tensor:
        """
        compute all body global position
        """
        joint_global_quat: torch.Tensor = self.diff_motion.joint_orientation.view(-1, 4)  # (num_frame * num_joint, 4)
        child_offset: torch.Tensor = self.bvh_child_body_offset.repeat(self.num_frames, 1, 1)  # (frame, joint, 3)
        child_offset: torch.Tensor = child_offset.view(-1, 3)  # (frame * joint, 3)
        offset: torch.Tensor = DiffQuat.quat_apply(joint_global_quat, child_offset)  # (frame * joint, 3)
        offset: torch.Tensor = offset.view(self.num_frames, -1, 3)  # (frame, joint, 3)
        self.child_body_position: torch.Tensor = offset + self.diff_motion.joint_position  # (frame, joint, 3)

        # TODO: here we should check body position
        return self.child_body_position

    def compute_com(self):
        """
        compute momentum and angular momentum with CoM
        """
        # reference to Karen Liu: A Quick Tutorial on Multibody Dynamics
        self.global_com: torch.Tensor = torch.sum(
            self.bvh_child_body_mass.view(1, -1, 1) *
            self.child_body_position, dim=1) / torch.as_tensor(self.total_mass)

        # compute angular momentum in global coordinate
        body_position: torch.Tensor = self.child_body_position  # (frame, joint, 3)
        body_orientation: torch.Tensor = self.diff_motion.joint_orientation  # (frame, joint, 3)
        body_linear_velo: torch.Tensor = self.child_body_velocity  # (frame, joint, 3)
        body_angular_velo: torch.Tensor = self.joint_angular_velo  # (frame, joint, 3)

        num_frames: int = self.num_frames
        num_joints: int = self.motion.num_joints
        dcm: torch.Tensor = DiffQuat.quat_to_matrix(body_orientation.view(-1, 4)).view(num_frames, num_joints, 3, 3)
        inertia: torch.Tensor = self.bvh_child_body_inertia.view(1, num_joints, 3, 3)
        inertia: torch.Tensor = dcm @ inertia @ torch.transpose(dcm, -1, -2)  # (num_frames, num_joints, 3, 3)
        angular_momentum: torch.Tensor = (inertia @ body_angular_velo[..., None]).view(num_frames, num_joints, 3)

        com_to_body: torch.Tensor = body_position - self.global_com.view(num_frames, 1, 3)  # (frame, joint, 3)
        body_linear_momentum: torch.Tensor = body_linear_velo * self.bvh_child_body_mass.view(1, -1, 1)  # (f, j, 3)
        angular_momentum: torch.Tensor = angular_momentum + torch.cross(com_to_body, body_linear_momentum, dim=-1)  # (frame, j, 3)
        angular_momentum: torch.Tensor = torch.sum(angular_momentum, dim=1)  # (frame, 3)

        # compute linear momentum of center of mass
        self.linear_momentum: Optional[torch.Tensor] = torch.sum(body_linear_momentum, dim=1)
        self.angular_momentum: Optional[torch.Tensor] = angular_momentum  # (frame, 3)

        # compute delta linear momentum
        acc_forward = True
        self.delta_linear_momentum: Optional[torch.Tensor] = PyTorchMathHelper.vec_diff(self.linear_momentum, acc_forward, self.motion.fps)
        self.delta_angular_momentum: Optional[torch.Tensor] = PyTorchMathHelper.vec_diff(self.angular_momentum, acc_forward, self.motion.fps)

    def compute_velocity(self):
        """
        compute required linear / angular velocity
        """
        vel_forward = False  # for compute velocity

        # 1. compute joint linear velocity
        self.joint_linear_velo: Optional[torch.Tensor] = self.diff_motion.compute_linear_velocity(vel_forward)
        # 2. compute body linear velocity
        self.child_body_velocity: Optional[torch.Tensor] = PyTorchMathHelper.vec_diff(self.child_body_position, vel_forward, self.motion.fps)
        # 3. compute body angular velocity, that is, joint global angular velocity
        self.joint_angular_velo: Optional[torch.Tensor] = self.diff_motion.compute_angular_velocity(vel_forward)
        # 4. compute joint linear acc
        self.joint_linear_acc: Optional[torch.Tensor] = PyTorchMathHelper.vec_diff(self.joint_linear_velo, True, self.motion.fps)
        # 5. compute joint angular acc
        self.joint_angular_acc: Optional[torch.Tensor] = PyTorchMathHelper.vec_diff(self.joint_angular_velo, True, self.motion.fps)

    def compute_fric_loss(self, dim: int = 0) -> torch.Tensor:
        """
        The friction force <= mu * normal force
        """
        # (frame, 4, 2)
        max_fric: torch.Tensor = self.contact_mu * torch.abs(self.contact_force_parameter[..., 1]) + torch.as_tensor(0.01)

        fric: torch.Tensor = torch.abs(self.contact_force_parameter[..., dim])  # (frame, 4, 2)

        # we need not to compute gradient of selection index
        fric_select = torch.as_tensor(fric > max_fric)  # (frame, 4, 2) in dtype == torch.bool
        if torch.any(fric_select):
            fric_loss: torch.Tensor = torch.mean((fric[fric_select] / max_fric[fric_select]))  # (1, )
        else:
            fric_loss: torch.Tensor = torch.as_tensor(0.0)
        return fric_loss

    def compute_kinematics_loss(self):
        # 1. the pose should be close to the ground truth 2d pose..
        joint_2d_position: torch.Tensor = None

    def visualize_debug_3d(self, vector3d: torch.Tensor, info=""):
        np_array: np.ndarray = vector3d.detach().cpu().clone().numpy()
        fig = plt.figure()
        vis_x = fig.add_subplot(311)
        vis_x.plot(np_array[..., 0])

        vis_y = fig.add_subplot(312)
        vis_y.plot(np_array[..., 1])

        vis_z = fig.add_subplot(313)
        vis_z.plot(np_array[..., 2])

        plt.title(info)
        plt.show()

    def calc_require_force(self) -> torch.Tensor:
        """
        compute required force on CoM
        """
        # self.visualize_linear_momentum()
        # self.visualize_debug_3d(self.delta_angular_momentum, "delta angular momentum")
        # exit(0)
        require_force: torch.Tensor = operator.sub(
            self.delta_linear_momentum,
            torch.as_tensor(self.total_mass * self.gravity).view(1, 3)
        )  # (frame, joint, 3)
        return require_force

    def compute_contact_force_loss(
        self,
        loss_dict_input: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Only optimize with contact force, s.t. the physics loss is minimized.
        """
        if loss_dict_input is None:
            loss_dict_input = {}

        # 1. loss of friction cone
        fric_x_loss: torch.Tensor = self.compute_fric_loss(0)
        fric_z_loss: torch.Tensor = self.compute_fric_loss(2)

        # # 2. loss of CoM
        # # check OK
        require_force: torch.Tensor = self.calc_require_force()
        contact_global_force: torch.Tensor = self.contact_global_force.view(self.num_frames, -1, 3)  # (frame, j, 3)
        contact_total_force: torch.Tensor = torch.sum(contact_global_force, dim=1)  # (frame, 3)
        linear_momentum_loss: torch.Tensor = torch.mean(torch.linalg.norm(require_force - contact_total_force, dim=-1))  # (1,)

        # 3. loss of angular momentum
        require_torque: torch.Tensor = self.delta_angular_momentum  # (frame, 3)
        contact_global_pos: torch.Tensor = self.contact_global_pos.view(self.num_frames, -1, 3)  # (f, 4 * 2, 3)
        contact_rel_pos: torch.Tensor = contact_global_pos - self.global_com.view(self.num_frames, 1, 3) # (f, 4 * 2, 3)
        contact_total_torque = torch.sum(torch.cross(contact_rel_pos, contact_global_force, dim=-1), dim=1)  # (f, 3)
        ang_momentum_loss: torch.Tensor = torch.mean(torch.linalg.norm(contact_total_torque - require_torque, dim=-1))  # (1,)
        # ang_momentum_loss = functional.l1_loss(contact_total_torque, require_torque)

        # 4. y component of contact force should >= 0
        contact_y_index = torch.as_tensor(self.contact_force_parameter[..., 1] < 0)
        if torch.any(contact_y_index):
            contact_force_y_loss = torch.mean(self.contact_force_parameter[..., 1][contact_y_index] ** 2)
        else:
            contact_force_y_loss = torch.as_tensor(0.0)

        # 5. penality of contact
        contact_label: torch.Tensor = self.contact_label.view(self.num_frames, -1)  # (frame, 4 * 2)
        contact_penality_pos_loss = torch.sum(contact_label * contact_global_pos[..., 1] ** 2)  # (1,)
        contact_global_velo = self.contact_global_velo.view(self.num_frames, -1, 3)  # (frame, 4 * 2, 3)
        contact_penality_vel_loss = torch.sum(torch.linalg.norm(contact_label[..., None] * contact_global_velo[..., [0, 2]], dim=-1))  # (1,)

        loss_dict = {
            "friction": self.loss_weight.w_friction * (fric_x_loss + fric_z_loss),
            "linear_momentum": self.loss_weight.w_linear_momentum * linear_momentum_loss,
            "angular_momentum": self.loss_weight.w_angular_momentum * ang_momentum_loss,
            "contact_force_y_loss": self.loss_weight.w_friction * contact_force_y_loss,
            "contact_penality_pos": self.loss_weight.w_contact_penality_pos * contact_penality_pos_loss,
            "w_contact_penality_vel": self.loss_weight.w_contact_penality_vel * contact_penality_vel_loss,
        }

        return {**loss_dict_input, **loss_dict}

    @staticmethod
    def get_total_loss_by_dict(
        loss_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        each loss item is saved in a dict
        """
        total_loss = torch.as_tensor(0.0, dtype=torch.float64)  # compute total loss
        for key, value in loss_dict.items():
            total_loss += value
        return total_loss

    def compute_loss(self) -> Dict[str, torch.Tensor]:
        # contact_loss_dict = self.compute_contact_force_loss()

        contact_global_pos: torch.Tensor = self.contact_global_pos.view(self.num_frames, -1, 3)  # (f, 4 * 2, 3)
        # 5. avoid contact height < 0
        contact_height: torch.Tensor = contact_global_pos[..., 1]  # (frame, 4 * 2)
        height_le_label = contact_height < 0
        if torch.any(height_le_label):
            contact_height_avoid_loss: torch.Tensor = torch.mean((1e4 * contact_height[height_le_label]) ** 2)  # (1,)
        else:
            contact_height_avoid_loss = torch.as_tensor(0.0, dtype=torch.float64)
        # contact_height_avoid_loss = torch.mean(torch.log(1e-1 + contact_height[contact_height < 1e-3]))

        # 6. smooth term of the motion
        linear_acc_smooth_loss: torch.Tensor = torch.mean(self.joint_linear_acc ** 2)  # (1,)
        angular_acc_smooth_loss: torch.Tensor = torch.mean(self.joint_angular_acc ** 2)  # (1,)

        # 7. the optimized result should be close to reference motion
        # We can compute difference between 6d vector or axis angle directly.
        # check OK.
        # reference_loss_rotate = functional.mse_loss(self.rotate_inputs_parameter, self.rotate_gt)  # (1,)
        # reference_loss_root = functional.mse_loss(self.root_pos_parameter, self.root_pos_gt)  # (1,)
        dummy_3d, camera_2d = TorchMotionProjection.torch_proj_to_2d(self.diff_motion.joint_position, self.gt_data.camera_param)
        loss_2d = functional.mse_loss(camera_2d, self.gt_data.pos2d, reduction="sum")

        # 8. some loss etc

        # sum all of loss term
        loss_dict = {
            "height_avoid": self.loss_weight.w_height_avoid * contact_height_avoid_loss,
            "smooth_linear_acc": self.loss_weight.w_smooth_linear_acc * linear_acc_smooth_loss,
            "smooth_angular_acc": self.loss_weight.w_smooth_angular_acc * angular_acc_smooth_loss,
            "reference_2d": self.loss_weight.w_reference_2d * loss_2d,
        }
        # loss_dict.update(contact_loss_dict)

        return loss_dict

    def foot_geom_info(self, body_name: str, joint_name: str):
        """
        compute end site by geometry info
        Note: when std-human is at T-Pose, rotation of all of joints and bodies are zero.
        So, we need not to consider joint or body rotation here.
        """
        body_index: int = self.character_body_names.index(body_name)
        joint_index: int = self.character_joint_names.index(joint_name)
        joint: ode.Joint = self.character.joints[joint_index]
        joint_anchor: np.ndarray = joint.getAnchorNumpy()  # (3,)
        heel_body: ode.Body = self.character.bodies[body_index]
        body_pos: np.ndarray = heel_body.PositionNumpy  # (3,)
        offset: np.ndarray = body_pos - joint_anchor  # from joint anchor to body center (3,)
        heel_geom: ode.GeomBox = list(heel_body.geom_iter())[0]  # len == 1
        geom_length: np.ndarray = np.array(heel_geom.getLengths())  # (3,)
        half_len: np.ndarray = 0.5 * geom_length  # (3,)

        return offset, half_len

    def geom_handle_heel(self, body_name: str, joint_name: str) -> np.ndarray:
        # for std-human, heel and toe are box
        offset_, half_len_ = self.foot_geom_info(body_name, joint_name)
        contact_pos_1_ = offset_ + np.array([half_len_[0], -half_len_[1], -half_len_[2]])
        contact_pos_2_ = offset_ + np.array([-half_len_[0], -half_len_[1], -half_len_[2]])
        return np.concatenate([contact_pos_1_[None, :], contact_pos_2_[None, :]], axis=0)[None, ...]  # (1, 2, 3)

    def geom_handle_toe(self, body_name: str, joint_name: str) -> np.ndarray:
        # for std-human, heel and toe are box
        offset_, half_len_ = self.foot_geom_info(body_name, joint_name)
        contact_pos_1_ = offset_ + np.array([half_len_[0], -half_len_[1], half_len_[2]])
        contact_pos_2_ = offset_ + np.array([-half_len_[0], -half_len_[1], half_len_[2]])
        return np.concatenate([contact_pos_1_[None, :], contact_pos_2_[None, :]], axis=0)[None, ...]  # (1, 2, 3)

    def compute_foot_contact_end_site(self):
        # This operation should be done at T-Pose
        self.character.load_init_state()

        l_heel_offset: np.ndarray = self.geom_handle_heel(self.l_heel_body_name, self.l_heel_joint_name)  # (1, 2, 3)
        r_heel_offset: np.ndarray = self.geom_handle_heel(self.r_heel_body_name, self.r_heel_joint_name)  # (1, 2, 3)

        l_toe_offset: np.ndarray = self.geom_handle_toe(self.l_toe_body_name, self.l_toe_joint_name)  # (1, 2, 3)
        r_toe_offset: np.ndarray = self.geom_handle_toe(self.r_toe_body_name, self.r_toe_joint_name)  # (1, 2, 3)

        # in shape (4, 2, 3)
        end_sites = np.concatenate([l_heel_offset, l_toe_offset, r_heel_offset, r_toe_offset], axis=0)
        self.end_site_offset = torch.from_numpy(end_sites)  # (4, 2, 3)

        # get parent index in mocap data
        end_names = [self.l_heel_joint_name, self.l_toe_joint_name, self.r_heel_joint_name, self.r_toe_joint_name]
        self.end_site_parent = [self.motion.joint_names.index(name) for name in end_names]  # (4,)

    def compute_contact_global_position(self) -> torch.Tensor:
        """
        Compute global position of end sites by parent joint position and orientation
        """
        ret_shape = (self.num_frames, len(self.end_site_parent), self.body_contact_count, 3)
        ret_tensor: torch.Tensor = torch.zeros(ret_shape, dtype=torch.float64)
        for parent_count, parent_joint_index in enumerate(self.end_site_parent):
            parent_quat = self.diff_motion.joint_orientation[:, parent_joint_index, :].contiguous()
            for contact_count in range(self.body_contact_count):
                local_offset: torch.Tensor = self.end_site_offset[parent_count, contact_count]  # (3,)
                local_offset_ext: torch.Tensor = local_offset.repeat(self.num_frames, 1)  # (num frames, 3)
                global_offset: torch.Tensor = DiffQuat.quat_apply(parent_quat, local_offset_ext)  # (num frames, 3)

                # here we should also consider parent root global position
                global_offset: torch.Tensor = global_offset + self.diff_motion.joint_position[:, parent_joint_index, :]
                ret_tensor[:, parent_count, contact_count, :] = global_offset

        ret_tensor: torch.Tensor = ret_tensor.contiguous()
        # ret_tensor: torch.Tensor = ret_tensor + None
        self.contact_global_pos: torch.Tensor = ret_tensor

        return self.contact_global_pos

    def compute_2d_proj_loss(self):
        pass

    def handle_contact(self):
        # 1. compute global endsite position
        self.compute_contact_global_position()

        # 2. compute global force (that is, convert contact force from facing coordinate to global coordinate)
        root_quat: torch.Tensor = self.diff_motion.joint_rotation[:, 0, :]
        self.facing_quat_y, _ = PyTorchMathHelper.y_decompose(root_quat)
        self.contact_global_force: Optional[torch.Tensor] = torch.zeros((self.num_frames, len(self.end_site_parent), 2, 3), dtype=torch.float64)
        if torch.any(self.contact_force_parameter != 0):
            for parent_count, parent_joint_index in enumerate(self.end_site_parent):
                for contact_count in range(self.body_contact_count):
                    local_contact: torch.Tensor = self.contact_force_parameter[:, parent_count, contact_count, :]
                    global_contact: torch.Tensor = DiffQuat.quat_apply(self.facing_quat_y, local_contact)
                    self.contact_global_force[:, parent_count, contact_count] = global_contact

        # 3. compute global torque
        contact_to_com: torch.Tensor = self.contact_global_pos - self.global_com.view(self.num_frames, 1, 1, 3)
        self.contact_global_torque: torch.Tensor = torch.cross(contact_to_com, self.contact_global_force)

        # 4. compute contact label via y component
        half = torch.as_tensor(0.5)
        self.contact_label: torch.Tensor = half + half * torch.tanh(
            torch.as_tensor(self.loss_weight.contact_label_k1) * torch.abs(self.contact_force_parameter[..., 1]) -\
            self.loss_weight.contact_label_k2)

        # 5. compute linear velocity of contact position
        self.contact_global_velo = PyTorchMathHelper.vec_diff(self.contact_global_pos, True, self.motion.fps)

    def compute_required_data(self):
        self.clear()  # clear previous middle tensors
        # set optimize parameter to PyTorchMotionData
        self.diff_motion.load_rot_trans(self.motion)
        quat_shape = (self.num_frames, len(self.joint_optim_index), 4)
        if self.rotate_type == RotateType.Vec6d:
            rotate_quat: torch.Tensor = DiffQuat.vec6d_to_quat(self.rotate_inputs_parameter.view(-1, 3, 2)).view(quat_shape)
        elif self.rotate_type == RotateType.AxisAngle:
            rotate_quat: torch.Tensor = DiffQuat.quat_from_rotvec(self.rotate_inputs_parameter.view(-1, 3)).view(quat_shape)
        else:
            raise NotImplementedError

        self.diff_motion.set_parameter(self.root_pos_parameter, rotate_quat, self.joint_optim_index)

        # forward kinematics
        self.diff_motion.recompute_joint_global_info()

        self.compute_child_body_pos_fast()
        self.compute_velocity()
        self.compute_com()
        self.handle_contact()

    def save(self):
        # save the middle result.
        result = {
            "epoch": self.epoch,
            "rotate_parameter": self.rotate_inputs_parameter.data,
            "contact_force_parameter": self.contact_force_parameter.data,
            "root_pos_parameter": self.root_pos_parameter.data,
            "optimizer": self.optim.state_dict()
        }
        torch.save(result, self.dump_fname)
        print(f"save dump file to {self.dump_fname}")

    def load(self):
        if os.path.exists(self.dump_fname):
            result: Dict[str, Any] = torch.load(self.dump_fname)
            self.epoch: int = result["epoch"]
            with torch.no_grad():
                self.contact_force_parameter.data = result["contact_force_parameter"]
            print(f"Load from {self.dump_fname}")
        else:
            print(f"dumping result not exist..ignore..")

    def export_to_bvh(self):
        """
        Export current diff rotation to bvh file..
        """
        motion: MotionData = self.diff_motion.export_to_motion_data()
        fname: str = os.path.join(self.dump_dir, f"epoch-{self.epoch}.bvh")
        BVHLoader.save(motion, fname)
        print(f"save epoch {self.epoch} to {fname}")

    def closure(self):
        self.optim.zero_grad()
        # compute required data
        self.clear()
        self.compute_required_data()

        # compute and print loss
        self.loss_dict: Dict[str, torch.Tensor] = self.compute_loss()
        # self.loss_dict: Dict[str, torch.Tensor] = self.compute_contact_force_loss()
        self.loss_dict["total_loss"] = self.get_total_loss_by_dict(self.loss_dict)

        total_loss: torch.Tensor = self.loss_dict["total_loss"]
        total_loss.backward()

        if self.rotate_type == RotateType.AxisAngle:
            self.rotate_inputs_parameter.grad[:, self.knee_optim_index, 1:] = 0

        # if self.contact_force_parameter.grad is not None:
        #    self.contact_force_parameter.grad *= 2
        # if self.root_pos_parameter.grad is not None:
        #    self.root_pos_parameter.grad /= 500
        # if self.rotate_inputs_parameter.grad is not None:
        #    self.rotate_inputs_parameter.grad /= 100

        return total_loss

    def train(self, export_to_bvh: bool = False):
        # first, we can optimize contact force term only (lbfgs optimizer for 50 epoches, with lr = 5e-2)
        # then, optimize total variables...
        while self.epoch < self.max_epoch:
            if self.run_as_sub_thread:
                self.initialize_bvh_target_setter()
                ServerThreadHandle.sub_thread_wait_for_run()

            # self.closure()
            # self.optim.step()
            self.optim.step(self.closure)  # Optimize
            # print(self.epoch, self.loss_dict["total_loss"])
            self.export_to_bvh()
            self.epoch += 1
            if self.run_as_sub_thread:
                ServerThreadHandle.sub_thread_run_end()
            if True:
                # self.save()
                print(f"\n=========At epoch {self.epoch}=============")
                for key, value in self.loss_dict.items():
                    print(f"{key}: {value.item()}")
            if self.epoch % 10 == 0:
                self.save()
            # input()

        self.save()
        print(f"After training for {self.epoch} epoches.")

    def extract_current_state(self, frame: int):
        # set current kinematics character state to ODE Character
        self.export_target_set.set_character_byframe(frame)

        if self.contact_global_pos is not None and self.contact_global_force is not None:
            contact_pos: Optional[np.ndarray] = self.contact_global_pos[frame].detach().clone().numpy().reshape((-1, 3))
            contact_force: np.ndarray = self.contact_global_force[frame].detach().clone().numpy().reshape((-1, 3))
            contact_label = self.contact_label[frame].detach().clone().numpy().reshape(-1)
            contact_info = SceneContactInfo(contact_pos, contact_force, contact_label=contact_label)
            self.scene.contact_info = contact_info
        else:
            self.scene.contact_info = None
