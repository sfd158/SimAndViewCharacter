'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''

from argparse import ArgumentParser, Namespace
from enum import IntEnum
from mpi4py import MPI
import numpy as np
import os
import pickle
import platform
import torch
from torch.nn import functional as F
from typing import Any, List, Tuple, Dict, Optional, Union
from tensorboardX import SummaryWriter

from VclSimuBackend.Common.Helper import Helper
from VclSimuBackend.Common.MathHelper import MathHelper, RotateType
from VclSimuBackend.DiffODE.DiffHinge import DiffHingeInfo
from VclSimuBackend.ODESim.BodyInfoState import BodyInfoState

from VclSimuBackend.ODESim.Saver.CharacterToBVH import CharacterTOBVH
from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
from VclSimuBackend.ODESim.Loader.JsonCharacterLoader import JsonCharacterLoader
from VclSimuBackend.ODESim.ODEScene import ODEScene
from VclSimuBackend.ODESim.ODECharacter import ODECharacter
from VclSimuBackend.ODESim.TargetPose import TargetPose
from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter

from VclSimuBackend.Render.Renderer import RenderWorld
from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import DirectTrajOptBVH, SamHlp
from VclSimuBackend.Samcon.SamconTargetPose import SamconTargetPose
from VclSimuBackend.Samcon.OptimalGait.ContactWithKinematic import ContactLabelExtractor
from VclSimuBackend.Samcon.OptimalGait.ContactPlan import ContactPlan

from VclSimuBackend.DiffODE.config import ContactConfig
from VclSimuBackend.DiffODE.MocapImport import MocapImport
from VclSimuBackend.DiffODE.Build import BuildFromODEScene
from VclSimuBackend.DiffODE.DiffContact import DiffContactInfo
from VclSimuBackend.DiffODE.DiffODEWorld import DiffODEWorld
from VclSimuBackend.DiffODE.DiffFrameInfo import DiffFrameInfo
from VclSimuBackend.Utils.Camera.CameraNumpy import CameraParamNumpy
from VclSimuBackend.Utils.Camera.CameraPyTorch import CameraParamTorch
from VclSimuBackend.Utils.CharacterStateExtractor import generate_contact_flag
from VclSimuBackend.Utils.Evaluation import calc_motion_mpjpe

from VclSimuBackend.pymotionlib.MotionData import MotionData
from VclSimuBackend.pymotionlib.PyTorchMotionData import PyTorchMotionData
from VclSimuBackend.pymotionlib import BVHLoader

fdir = os.path.dirname(__file__)
cpu_device = torch.device("cpu")
comm: MPI.Intracomm = MPI.COMM_WORLD
comm_rank: int = comm.Get_rank()
comm_size: int = comm.Get_size()
is_windows: bool = "Windows" in platform.platform()


def clear_folder():
    if comm_rank == 0:
        list_dir = os.listdir(fdir)
        num_deleted = 0
        for fname in list_dir:
            cat_fname = os.path.join(fdir, fname)
            if "OptimizeEachFrame.ckpt" in fname:
                os.remove(cat_fname)
                num_deleted += 1
            elif "rank0-2022" in fname:
                os.removedirs(cat_fname)
                num_deleted += 1
            else:
                continue
        print(f"deleted {num_deleted} tmp files")
    exit(0)


def disable_phys_loss(args: Namespace) -> Namespace:
    args.w_loss_pos = 0
    args.w_loss_rot = 0
    args.w_loss_velo = 0
    args.w_loss_angvel = 0
    return args


def load_phys_loss(args: Namespace, w_loss_pos: float, w_loss_rot: float, w_loss_velo: float, w_loss_angvel: float) -> Namespace:
    args.w_loss_pos = w_loss_pos
    args.w_loss_rot = w_loss_rot
    args.w_loss_velo = w_loss_velo
    args.w_loss_angvel = w_loss_angvel
    return args


def save_phys_loss(args: Namespace) -> Tuple[float, float, float, float]:
    return (args.w_loss_pos, args.w_loss_rot, args.w_loss_velo, args.w_loss_angvel)


def only_use_phys_loss(args: Namespace):
    args.w_root_close = 0
    args.w_joint_close = 0
    args.w_pd_target_close = 0
    args.w_root_smooth = 0
    args.w_joint_smooth = 0
    args.w_pd_target_smooth = 0
    args.w_loss_2d = 0
    return args


class ContactPlanMode(IntEnum):
    NoUse = 0
    Kinematic = 1
    Greedy = 2
    MCMC = 3
    SMC = 4


def parse_args() -> Namespace:
    """
    parse arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--index_t", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--config_fname", type=str, default=os.path.join(fdir, "../../../Tests/CharacterData/SamconConfig-duplicate.json"))
    parser.add_argument("--process_fname", type=str,
        default=r"D:\song\documents\GitHub\ode-scene\wild-sitting-floor10\network-output.bin")
        # default=os.path.join(fdir, "../../../Tests/CharacterData/Samcon/0/network-output.bin"))
    parser.add_argument("--output_fdir", type=str, default=None)
    parser.add_argument("--opt_result_sub_dir", type=str, default="opt_result", help="save for samcon result")
    parser.add_argument("--checkpoint_fname", type=str, default="")
    parser.add_argument("--save_num_epoch", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--simulation_fps", type=float, default=50)
    parser.add_argument("--phys_loss_func", type=str, default="l2")  # smooth l1
    parser.add_argument("--phys_loss_num_forward", type=int, default=1)
    parser.add_argument("--use_predicted_target_pose", action="store_true", default=True)

    parser.add_argument("--contact_plan_mode", type=str, default="Kinematic", choices=ContactPlanMode._member_names_)

    parser.add_argument("--contact_plan_use_mcmc", action="store_true", default=True)
    parser.add_argument("--contact_plan_mcmc_frame", type=int, default=200,
        help="we can pre-compute local phys loss on one frame, and sample the contact label with prob from local phys loss."
    )
    parser.add_argument("--contact_plan_mcmc_number", type=int, default=100, help="Total sample sequence number")
    parser.add_argument("--contact_plan_mcmc_opt_step", type=int, default=10, help="")

    # parser.add_argument("--contact_plan_change_num", type=int, default=3)
    # parser.add_argument("--contact_plan_group_num", type=int, default=1
    parser.add_argument("--root_y_offset", type=float, default=0,
        help="TODO: We can add a global offset on y component for the whole body motion.."
    )

    parser.add_argument("--add_contact_height_eps", type=float, default=1, help="we should not create contact with height >= add_contact_height_eps")
    parser.add_argument("--use_previous_contact", action="store_true", default=False, help="use previous contact in optimization..")
    parser.add_argument("--use_contact_plan", action="store_true", default=True)
    parser.add_argument("--optimize_num_for_plan", type=int, default=0,
        help="For contact planning, we can optimize for several steps only with phys loss, and choose the contact label by optimized phys loss. "
        "The optimization result will be dropped after optimization. We only takes the contact label."
        "0 is use phys loss directly, without optimization."
    )

    parser.add_argument("--use_inf_friction", action="store_true", default=False)
    parser.add_argument("--use_ball_as_contact", action="store_true", default=True)
    parser.add_argument("--contact_cfm_normal", type=float, default=1e-10, help="use soft contact model for optimization")
    parser.add_argument("--contact_cfm_tan", type=float, default=1e-10, help="use soft contact..")
    parser.add_argument("--contact_erp", type=float, default=0.2, help="use soft contact model for optimization")
    parser.add_argument("--use_erp_to_floor", action="store_true", default=True)
    parser.add_argument("--contact_erp_to_floor", type=float, default=0.2)
    parser.add_argument("--use_cio", action="store_true", default=False)
    parser.add_argument("--also_optimize_velo", action="store_true", default=True)
    parser.add_argument("--optimize_velo_in_maximal", action="store_true", default=True)

    parser.add_argument("--ground_truth_pickle_fname", type=str,
        default="",
        # default=r"z:/GitHub/ode-develop/Tests/CharacterData/Samcon/0/S11-S11-Walking-mocap-100.pickle",
        # default=os.path.join(fdir, "../../../Tests/CharacterData/Samcon/0/network-output.bin"),
        help="use ground truth samcon data, for extract contact label.")

    parser.add_argument("--contact_eps", type=float, default=0.05)  # generate contact label
    parser.add_argument("--collision_detection_each_epoch", action="store_true", default=False)
    parser.add_argument("--use_lbfgs", action="store_true", help="use L-BFGS optimizer", default=False)
    parser.add_argument("--use_summary_writer", action="store_true", default=True)

    parser.add_argument("--rotate_type", type=str, default="Vec6d", choices=RotateType._member_names_)
    parser.add_argument("--max_epoch", type=int, default=440)
    parser.add_argument("--optimize_root_epoch", type=int, default=400, help="first, optimize the root position parameter only (maybe with large initial learning rate..)")
    parser.add_argument("--phys_loss_epoch", type=int, default=400)

    parser.add_argument("--lr_decay_epoch", type=int, default=100)
    parser.add_argument("--lr_decay_ratio", type=float, default=0.9)

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # 1e-3
    parser.add_argument("--reject_lr_decay", type=float, default=0.7)
    parser.add_argument("--w_loss_com", type=float, default=0) # 1e4, help="External force (contact force and gravity) and change of center of mass should match.")
    parser.add_argument("--w_loss_pos", type=float, default=5e4) # 2e6, help="The simulation result at time t + 1 should be close to kinematic state at t + 1. This is for body pos")
    parser.add_argument("--w_loss_rot", type=float, default=2e4) # 2e5, help="The simulation result at time t + 1 should be close to kinematic state at t + 1.")
    parser.add_argument("--w_loss_velo", type=float, default=5e-1) # 1e1, help="The simulation result at time t + 1 should be close to kinematic state at t + 1.")
    parser.add_argument("--w_loss_angvel", type=float, default=5e-1) # 5, help="The simulation result at time t + 1 should be close to kinematic state at t + 1.")

    parser.add_argument("--w_root_close", type=float, default=1e1) # 5e1 * ratio, help="The root position should be close to initial solution")
    parser.add_argument("--w_joint_close", type=float, default=1e-1) # 5e2 * ratio, help="The joint rotation should be close to initial solution")
    parser.add_argument("--w_pd_target_close", type=float, default=1e0) # 1e4 * ratio, help="The PD target pose should be close to initial solution")
    parser.add_argument("--w_root_smooth", type=float, default=1e1) # 1e2) # 5e5 * ratio, help="The kinematic root position should be smooth.")
    parser.add_argument("--w_joint_smooth", type=float, default=2e0) # 1e2) # 2e4 * ratio, help="The kinematic joint rotation should be smooth.")
    parser.add_argument("--w_pd_target_smooth", type=float, default=1e0) # 1e2) # 5e2 * ratio, help="The pd target should be smooth.")
    parser.add_argument("--w_loss_2d", type=float, default=1e-1) # 5e4 * ratio, help="The kinematic 3d pose should match the projection 2d pos")
    parser.add_argument("--w_contact_height", type=float, default=1e2) # 1e1 * ratio, help="the contact height should be zero, for physics plausible")
    parser.add_argument("--w_contact_horizontal_velo", type=float, default=1e2, help="the contact body should not move at x, z axis.")
    parser.add_argument("--w_contact_kine", type=float, default=5e5, help="used in optimize root")
    parser.add_argument("--w_contact_dyn", type=float, default=5e2, help="used in dynamic optimization")

    parser.add_argument("--use_hinge_limit", action="store_true", default=False)
    parser.add_argument("--use_hard_hinge_limit", action="store_true", default=False)
    parser.add_argument("--soft_hinge_limit_loss", type=float, default=0,
        help="use hinge angle limit as soft constraint, rather than hard constraint"
    )
    parser.add_argument("--only_use_phys_loss", action="store_true", default=False)
    parser.add_argument("--disable_phys_loss", action="store_true", default=False)
    parser.add_argument("--disable_contact_loss", action="store_true", default=False)
    # parser.add_argument("--w_body_height", type=float, default=1, help="If the height of body < 0, we should add penality..")
    # parser.add_argument("--w_com_inside_contact", type=float, default=1, help="The center of mass should fall inside the contact")

    parser.add_argument("--pos_grad_clip", type=float, default=50,
        help="clip gradient for root position variable")
    parser.add_argument("--rot_grad_clip", type=float, default=10,
        help="clip gradient for rotation variable")

    parser.add_argument("--use_hack_contact", action="store_true", default=True)
    parser.add_argument("--print_log_info", action="store_true", default=True)

    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--only_vis_result", action="store_true")
    parser.add_argument("--clear", action="store_true")

    args: Namespace = parser.parse_args()

    if os.path.isdir(args.process_fname):
        args.process_fname = os.path.join(args.process_fname, "network-output.bin")

    if args.output_fdir is None:
        args.output_fdir = os.path.abspath(fdir)
    else:
        if comm_rank == 0:
            if not os.path.exists(args.output_fdir):
                os.makedirs(args.output_fdir, exist_ok=True)

    args.opt_result_sub_dir = os.path.join(args.output_fdir, args.opt_result_sub_dir)

    if args.phys_loss_num_forward > 1:
        args.use_ball_as_contact = False  # I don't think ball contact will work for multi frame
        # maybe inf friction is just OK for mult frame..?
        # and we should not use cio here..

    if not args.use_hinge_limit:
        args.use_hard_hinge_limit = False
        args.soft_hinge_limit_loss = 0

    if args.only_use_phys_loss:
        args = only_use_phys_loss(args)

    if args.disable_phys_loss:
        disable_phys_loss()

    if args.disable_contact_loss:
        args.w_contact_height = 0
        args.w_contact_horizontal_velo = 0

    # Helper.set_torch_seed()
    if comm_rank > 0:
        args.use_summary_writer = False
        args.print_log_info = False
        args.debug_mode = False

    args.contact_plan_mode = ContactPlanMode[args.contact_plan_mode]
    ContactPlan.add_contact_height_eps = args.add_contact_height_eps
    ContactConfig.use_inf_friction = args.use_inf_friction
    ContactConfig.erp_to_floor = args.use_erp_to_floor
    if args.use_erp_to_floor:
        args.contact_erp = args.contact_erp_to_floor
    DiffHingeInfo.enable_limit = args.use_hard_hinge_limit

    if args.clear:  # clear the result file
        clear_folder()

    return args


class OptimizeEachFrameBase:
    """
    The gradient will be detached at each time step.
    """

    def __init__(self, args: Namespace, scene: Optional[ODEScene] = None) -> None:
        self.args = args
        samhlp: SamHlp = SamHlp(args.config_fname)
        conf: Dict[str, Any] = samhlp.conf
        conf["inverse_dynamics"]["in_use"] = False
        conf["traj_optim"]["load_samcon_result"] = False

        if scene is None:
            self.scene: ODEScene = ODEScene()
            # Note: here hinge and ball CFM should not be modified. we should modify the contact cfm only..
            self.scene: ODEScene = DirectTrajOptBVH.load_scene_with_conf(conf, self.scene)
            # for debug, we can render the initial kinematics motion..
            scene_dict: Dict[str, Any] = pickle.load(open(conf["filename"]["world"], "rb"))
            character_dict: Dict[str, Any] = scene_dict["CharacterList"]["Characters"][0]
            self.ref_character: ODECharacter = JsonCharacterLoader(self.scene.world, self.scene.space).load(character_dict)
            self.ref_character.is_enable = False

        else:
            self.scene = scene
            self.ref_character = self.scene.characters[1]

        self.ref_character_gt = None
        self.scene.set_sim_fps(args.simulation_fps)
        self.scene.use_soft_contact = True
        self.scene.soft_cfm = args.contact_cfm_normal
        self.scene.soft_cfm_tan = args.contact_cfm_tan
        self.scene.soft_erp = args.contact_erp
        self.scene.characters = [self.scene.character0]

        if comm_size == 1:
            print(f"scene simulation fps = {self.scene.sim_fps}, CFM = {self.scene.world.CFM}")
        self.character: ODECharacter = self.scene.character0
        self.character_body_names: List[str] = self.character.body_info.get_name_list()
        self.num_body: int = len(self.character.bodies)
        self.to_bvh: CharacterTOBVH = CharacterTOBVH(self.character, self.scene.sim_fps)
        self.to_bvh.build_hierarchy()

        # print(f"num body = {len(self.character.bodies)}, num joints = {len(self.character.joints)}")
        ret = samhlp.load_inv_dyn_from_pickle(self.scene, self.character, args.process_fname, debug_print=False)
        self.motion: MotionData = ret[0]
        assert isinstance(self.motion, MotionData)
        if args.width is None:
            args.width = self.motion.num_frames
        self.raw_motion: MotionData = self.motion.sub_sequence(copy=True)
        self.motion_hierarchy_with_end: MotionData = self.motion.get_hierarchy(True)
        self.motion: MotionData = self.motion.remove_end_sites()
        self.motion_hierarchy: MotionData = self.motion.get_hierarchy(True)
        self.samcon_target: SamconTargetPose = ret[1]
        assert isinstance(self.samcon_target, SamconTargetPose)
        self.target: TargetPose = self.samcon_target.pose
        self.set_tar: SetTargetToCharacter = SetTargetToCharacter(self.character, self.target)

        self.diff_motion: PyTorchMotionData = PyTorchMotionData()
        self.diff_motion.build_from_motion_data(self.motion.get_hierarchy(), torch.float64)
        self.builder = BuildFromODEScene(self.scene)
        self.diff_world: DiffODEWorld = self.builder.build()
        self.body_mass: torch.Tensor = self.curr_frame.body_frame.const_info.mass.clone()
        self.tot_mass: torch.Tensor = self.curr_frame.body_frame.const_info.total_mass.clone()
        self.body_inertia: torch.Tensor = self.curr_frame.body_frame.const_info.init_inertia.clone()

        self.mocap_import: MocapImport = MocapImport(self.character, self.motion)
        self.character_to_bvh: np.ndarray = self.mocap_import.character_to_bvh
        assert np.all(np.array(self.character.get_joint_names(True)) == np.array(self.motion.joint_names)[self.character_to_bvh])

        invdyn_target: TargetPose = ret[2]
        assert isinstance(invdyn_target, TargetPose)
        if args.use_predicted_target_pose:  # note: here the order should follows the joint order in open dynamics engine..
            self.torch_target_quat: torch.Tensor = torch.as_tensor(invdyn_target.locally.quat, dtype=torch.float64)
            if comm_rank == 0:
                print(f"use predicted target pose", flush=True)
        else:
            self.torch_target_quat: torch.Tensor = torch.as_tensor(self.samcon_target.pose.all_joint_local.quat[:, 1:, :], dtype=torch.float64)
            if comm_rank == 0:
                print(f"use kinematic as target pose", flush=True)

        # Test: output the target quat
        # Here the pd target is correct..
        # test_motion = self.to_bvh.forward_kinematics(self.motion.joint_position[:, 0], self.motion.joint_rotation[:, 0], self.torch_target_quat.numpy())
        # BVHLoader.save(test_motion, "test_motion.bvh")
        # exit(0)

        self.confidence: np.ndarray = ret[3]
        assert isinstance(self.confidence, np.ndarray)
        self.torch_joint_pos_2d: torch.Tensor = torch.as_tensor(self.samcon_target.pose2d_unified.pos2d, dtype=torch.float64)
        self.torch_confidence: torch.Tensor = torch.as_tensor(self.confidence, dtype=torch.float64)

        # we can also visualize the samcon result in the same drawstuff window
        if os.path.exists(args.ground_truth_pickle_fname) and False:
            # extract the contact label using ground truth
            gt_pickle: List[BodyInfoState] = pickle.load(open(args.ground_truth_pickle_fname, "rb"))[76:-76:2]  # TODO
            self.contact_label: np.ndarray = generate_contact_flag(self.scene, self.character, gt_pickle)
            self.ref_gt_tarset = lambda frame_: self.ref_character_gt.load(gt_pickle[frame_]) if self.ref_character_gt is not None else None
            if comm_rank == 0:
                print(f"use ground truth contact label from file {args.ground_truth_pickle_fname}, shape = {self.contact_label.shape}")
        else:
            self.contact_label: np.ndarray = ret[4]
            # draw_foot_contact_plan(self.contact_label)
            if comm_rank == 0:
                print(f"use estimated contact label, shape = {self.contact_label.shape}")
            self.ref_gt_tarset = None

        assert isinstance(self.contact_label, np.ndarray)
        # here we should build the contact sequence from kinematic motion and the predicted contact label..
        self.extractor: ContactLabelExtractor = ContactLabelExtractor(self.scene, self.character)
        self.body_min_contact_h: torch.Tensor = torch.as_tensor(self.extractor.body_min_contact_h)
        self.contact_mess: List[List[int]] = [np.where(self.contact_label[frame] >= args.contact_eps)[0].tolist()
            for frame in range(self.contact_label.shape[0])]
        # self.render = RenderWorld(self.scene)
        # self.render.start()
        self.contact_mess: List[List[int]] = self.extractor.preprocess_contact_mess(self.contact_mess, self.target)

        # here we should refine the contact mess. That is, when one body maintains a contact, the higher neighbour should not maintain a contact..
        if not args.collision_detection_each_epoch:
            contact_info = self.extractor.handle_mocap(self.target, False, self.contact_label, args.contact_eps)
            self.contact_info_list: List[DiffContactInfo] = self.extractor.convert_to_diff_contact(*contact_info, self.target)
        else:
            self.contact_info_list = None
        if comm_rank == 0 and False:
            for frame, node in enumerate(self.contact_mess):
                print(frame, node, end="")
                contact_info = self.contact_info_list[frame]
                if contact_info is not None:
                    print(contact_info.body0_index.tolist())
                else:
                    print()

        self.camera_param: CameraParamNumpy = ret[5]
        assert isinstance(self.camera_param, CameraParamNumpy)
        self.camera_torch: CameraParamTorch = CameraParamTorch.build_from_numpy(self.camera_param, dtype=torch.float64)

        self.best_loss: float = float("inf")
        self.best_param: Optional[Dict[str, Any]] = None

        self.global_step: int = 0

        if args.use_summary_writer and comm_rank == 0:
            writer_fdir = os.path.join(args.output_fdir, f"rank{comm_rank}-{Helper.get_curr_time()}")
            self.writer: Optional[SummaryWriter] = SummaryWriter(writer_fdir)
        else:
            self.writer: Optional[SummaryWriter] = None

        if args.debug_mode and is_windows:  # render using Long Ge's Framework
            self.render: RenderWorld = RenderWorld(self.scene)
            self.render.start()  # visualize the character
        else:
            self.render: Optional[RenderWorld] = None

        # here we can also add ground truth result, for debug..
        with open(args.process_fname, "rb") as fin:
            process_result = pickle.load(fin)
        if "input_motion_gt" in process_result:
            gt_fname: str = os.path.join(os.path.dirname(args.process_fname), process_result["input_motion_gt"])
        else:
            gt_fname = ""
        if os.path.exists(gt_fname):
            self.gt_mocap: Optional[MotionData] = BVHLoader.load(gt_fname)
            self.gt_mocap: Optional[MotionData] = self.gt_mocap.resample(self.scene.sim_fps)
            if comm_rank == 0:
                init_facing_mpjpe = calc_motion_mpjpe(self.gt_mocap, self.motion)
                init_mpjpe = calc_motion_mpjpe(self.gt_mocap, self.motion, False)
                print(f"init facing mpjpe = {init_facing_mpjpe:.6f}, init mpjpe = {init_mpjpe:.6f}")
            self.gt_target: Optional[TargetPose] = BVHToTargetBase(self.gt_mocap, self.scene.sim_fps, self.character).init_target()
            self.gt_tar_set: Optional[SetTargetToCharacter] = SetTargetToCharacter(self.ref_character, self.gt_target)
            self.gt_mocap_cp: Optional[MotionData] = self.gt_mocap.sub_sequence(copy=True)
            self.gt_mocap: Optional[MotionData] = self.gt_mocap.remove_end_sites()
            # self.gt_global_joint: Optional[torch.Tensor] = torch.from_numpy(self.gt_mocap.joint_position)
            # self.gt_global_joint_velo: Optional[torch.Tensor] = torch.diff(self.gt_global_joint, dim=0)  # for evaluate the smooth loss in Physcap.
            # Note: the velocity of frame 0 will not be used in evaulation..only for index..
            # self.gt_global_joint_velo: Optional[torch.Tensor] = torch.cat([self.gt_global_joint_velo[None, 0], self.gt_global_joint_velo], dim=0)
        else:
            self.gt_mocap: Optional[MotionData] = None
            self.gt_mocap_cp: Optional[MotionData] = None
            self.gt_target: Optional[TargetPose] = None
            self.gt_tar_set: Optional[SetTargetToCharacter] = None
            self.gt_global_joint: Optional[torch.Tensor] = None
            self.gt_global_joint_velo: Optional[torch.Tensor] = None

        self.scene.characters = [self.scene.character0, self.ref_character]

        # if args.phys_loss_func in ["l1", "l1_loss"]:
        #     self.phys_loss_func = F.l1_loss
        # elif args.phys_loss_func in ["l2", "mse", "l2_loss", "mse_loss"]:
        #     self.phys_loss_func = F.mse_loss
        # elif args.phys_loss_func in ["smooth_l1", "smooth_l1_loss"]:
        #     self.phys_loss_func = F.smooth_l1_loss
        # else:
        #     raise NotImplementedError
        self.phys_loss_func = lambda x, y: torch.sum((x - y) ** 2)

    @property
    def curr_frame(self) -> DiffFrameInfo:
        return self.diff_world.curr_frame

    def forward_kinematics_np(
        self,
        root_pos_numpy: np.ndarray,
        joint_numpy: np.ndarray,
    ) -> MotionData:
        mocap = self.to_bvh.forward_kinematics(root_pos_numpy, joint_numpy[:, 0, :], joint_numpy[:, 1:, :])
        mocap: MotionData = self.to_bvh.insert_end_site(mocap)
        mocap.recompute_joint_global_info()
        return mocap

    def convert_contact_mess_to_ndarray(self, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        if start is None:
            start: int = 0
        if end is None:
            end: int = len(self.contact_mess)
        contact_mess: List[List[int]] = self.contact_mess[start:end]
        continuous_contact: np.ndarray = np.zeros((len(contact_mess), len(self.character.bodies)))
        for index, mess in enumerate(contact_mess):
            if mess is not None:
                continuous_contact[index, mess] = 1
        return continuous_contact

    def draw_contact_mess(self):
        vis_contact_plan: np.ndarray = self.convert_contact_mess_to_ndarray()
        draw_full_body_contact(vis_contact_plan, self.character_body_names, False, "full sequence")
        # here we can render in small length..
        tot_num = vis_contact_plan.shape[0] // 50
        for i in range(tot_num):
            draw_full_body_contact(vis_contact_plan[i * 50: (i + 1) * 50], self.character_body_names, False, f"start={50 * i}")


def draw_full_body_contact(contact_label: np.ndarray, body_names: List[str], render_all_body: bool = False, title_info: str = ""):
    """
    Note: for the whole body motion, we should not render the body without contact on the whole sequence
    """
    num_frame: int = contact_label.shape[0]
    num_body: int = len(body_names)
    assert num_body == contact_label.shape[1]
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    fig: Figure = plt.figure()
    if render_all_body:
        reduced_contact: np.ndarray = np.ones(num_body, dtype=np.int32)
    else:
        reduced_contact = np.sum(contact_label, axis=0)  # shape == (num body,)
        reduced_contact = np.where(reduced_contact > 0)[0]
    num_render_body: int = reduced_contact.size
    for render_idx in range(num_render_body):
        # how to render more than 9 lines...?
        sub_plot = fig.add_subplot(num_render_body, 1, render_idx + 1)
        body_index = reduced_contact[render_idx]
        plt.plot(contact_label[:, body_index])
        plt.title(f"{body_index}-{body_names[body_index]}")
    # for linux system, we should save the result as png file.
    plt.suptitle(title_info)
    plt.show()


def draw_foot_contact_plan(contact_label: np.ndarray, contact_label2: Optional[np.ndarray] = None):
    # 'rFoot', 'lFoot', 'rToes', 'lToes'
    sub_index = dict(
        r_foot_index = 7,
        l_foot_index = 8,
        r_toe_index = 9,
        l_toe_index = 10,
    )
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    fig: Figure = plt.figure()
    for index, (key, value) in enumerate(sub_index.items()):
        sub_plot = fig.add_subplot(411 + index)
        plt.title(key)
        plt.plot(contact_label[:100, value])
        if contact_label2 is not None:
            plt.plot(contact_label2[:100, value])

    plt.show()


if __name__ == "__main__":
    print(parse_args())
