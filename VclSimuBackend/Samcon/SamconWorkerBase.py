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

import enum
from mpi4py import MPI
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, List, Dict, Union, Any

from VclSimuBackend.Common.Helper import Helper

from .Loss.SamconLoss import SamconLoss
from .SamconTargetPose import SamconTargetPose
from .SamHlp import SamHlp

from ..ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from ..ODESim.Saver.CharacterToBVH import CharacterTOBVH
from ..ODESim.CharacterWrapper import CharacterWrapper, ODECharacter
from ..ODESim.ODEScene import ContactType, ODEScene, SimulationType
from ..ODESim.PDControler import DampedPDControler
from ..ODESim.TargetPose import SetTargetToCharacter, TargetPose


from ..Utils.Camera.Human36CameraBuild import CameraParamBuilder
from ..Utils.Camera.CameraNumpy import CameraParamNumpy


class CameraConfigType(enum.IntEnum):
    HUMAN_36 = 0


class WorkerInfo:
    """
    wrapper of mpi4py info
    """
    def __init__(self):
        self.comm: MPI.Intracomm = MPI.COMM_WORLD
        self.comm_size: int = self.comm.Get_size()
        self.worker_size: int = max(1, self.comm_size - 1)
        self.comm_rank: int = self.comm.Get_rank()
        self.node_name: str = MPI.Get_processor_name()


class WorkerMode(enum.IntEnum):
    STOP = 0
    IDLE = 1

    ODE_SIMU = 2  # ODE Simulation
    TRAJECTORY_OPTIM = 3  # trajectory optimization
    CMA = 4

    SEND_TARGET_NO_SCATTER = 16
    RECIEVE_TARGET = 17

    SET_LOSS_LIST_INDEX = 32
    ENABLE_DUPLICATE = 33


class CMAInitCovMode(enum.IntEnum):
    IDENTITY = 0
    FIXED = 1


class CMAInfo:
    def __init__(self, conf: Dict, character: ODECharacter):
        """
        param:
        conf: configure dict
        character: character
        """
        self.conf: Dict[str, Any] = conf
        worker_cma: Dict[str, Union[float, int, bool, Dict]] = conf["worker_cma"]

        self.dup_count: Optional[int] = None
        # minimal iteration at each control fragment in a trial. In Improved Samcon 2015, it was set to 5
        self.min_iteration: Optional[int] = None

        # maximal iteration at each control fragment in a trial. In Improved Samcon 2015, it was set to 20
        self.iteration: Optional[int] = None

        # initial cma sigma. In Improved Samcon 2015, it was set to 0.1
        self.cost_exponent: Optional[int] = None
        self.first_trial_iteration: Optional[int] = None
        self.init_sigma_list: Optional[List[float]] = None
        self.init_sigma: Optional[float] = None

        self.sigma_ratio: Optional[List[float]] = [1.0]

        self.cma_small_eps: Optional[float] = None
        self.cma_large_eps: Optional[float] = None

        self.com_good_ratio: Optional[float] = None
        self.com_err_ratio: Optional[float] = None
        self.com_y_err_ratio: Optional[float] = None

        self.change_cma_param(worker_cma)

        self.cma_small_eps: float = worker_cma["cma_small_eps"]
        self.cma_large_eps: float = worker_cma["cma_large_eps"]
        self.loss_window: int = worker_cma["loss_window"]
        # size of sliding window. In Improved Samcon 2015, it was set to 50
        self.sliding_window: int = worker_cma["sliding_window"]

        self.start_decay: int = worker_cma["start_decay"]
        self.stop_decay: int = worker_cma["stop_decay"]

        self.direct_forward: bool = worker_cma["direct_forward"]

        self.no_falldown_forward: bool = worker_cma["no_falldown_go_forward"]

        dup_info = worker_cma.get("duplicate_input", None)
        if not (dup_info and dup_info["in_use"]):
            self.dup_count: Optional[int] = 1

        self.cov_mode = self.get_init_sample_cov_mat(character, conf)

        # for fine-tune
        self.piece_num = 3

        self.use_cma: bool = worker_cma["use_cma"]
        if not self.use_cma:
            self.dup_count = 1
            self.first_trial_iteration = 0
            self.min_iteration = 0
            self.iteration = 0

    def change_cma_param(self, worker_cma: Dict[str, Any]):
        self.cost_exponent: Optional[int] = worker_cma.get("cost_exponent", self.cost_exponent)
        self.first_trial_iteration = worker_cma.get("first_trial_iteration", self.first_trial_iteration)

        # minimal iteration at each control fragment in a trial. In Improved Samcon 2015, it was set to 5
        self.min_iteration = worker_cma.get("cma_min_iteration", self.min_iteration)

        # maximal iteration at each control fragment in a trial. In Improved Samcon 2015, it was set to 20
        self.iteration: int = worker_cma.get("cma_iteration", self.iteration)

        self.sigma_ratio: Optional[List[float]] = worker_cma.get("sigma_ratio", self.sigma_ratio)
        # print(self.sigma_ratio)  # for debug

        # initial cma sigma. In Improved Samcon 2015, it was set to 0.1
        self.init_sigma_list: float = worker_cma.get("init_cma_sigma_list", self.init_sigma_list)
        if self.init_sigma_list is not None:
            self.init_sigma = self.init_sigma_list[0]

        self.com_good_ratio: float = worker_cma.get("com_good_ratio", self.com_good_ratio)
        self.com_err_ratio: float = worker_cma.get("com_err_ratio", self.com_err_ratio)
        self.com_y_err_ratio: float = worker_cma.get("com_y_err_ratio", self.com_y_err_ratio)

        self.dup_count: float = worker_cma.get("dup_count", self.dup_count)

    def get_init_sample_cov_mat(self, character: ODECharacter, conf: Dict[str, Any]) -> np.ndarray:
        """
        return in shape (num joint, 3)
        """
        cov_mode: CMAInitCovMode = CMAInitCovMode[conf["worker_cma"]["cov_mode"]]
        samwin_dict: Dict[str, List[float]] = conf["character"]["sample_window"]
        if cov_mode == CMAInitCovMode.IDENTITY:
            result: np.ndarray = np.ones((len(character.joints), 3), dtype=np.float64)
        elif cov_mode == CMAInitCovMode.FIXED:
            result_list = []
            for jidx, jname in enumerate(character.joint_info.joint_names()):
                if jname in samwin_dict:
                    result_list.append(samwin_dict[jname])
            result = np.array(result_list)
        else:
            raise ValueError

        character.joint_info.sample_win = result
        return cov_mode


class LoadTargetPoseMode(enum.IntEnum):
    BVH_MOCAP = 0  # compute target pose and inverse dynamics result from bvh mocap data
    PICKLE_TARGET = 1  # target pose and inverse dynamics target pose is precomputed and saved in the pickle file


class SamconWorkerBase(CharacterWrapper):
    def __init__(
        self,
        samhlp: SamHlp,
        worker_info: Optional[WorkerInfo],
        scene: Optional[ODEScene] = None,
        sim_character: Optional[ODECharacter] = None,
    ):
        """
        param:
        conf_name: configuration file name
        worker_info: MPI info
        """
        super(SamconWorkerBase, self).__init__(sim_character)
        self.conf: Dict[str, Any] = samhlp.conf
        self.samhlp = samhlp

        if worker_info is None:
            worker_info = WorkerInfo()

        self.worker_info: WorkerInfo = worker_info

        # Load Scene and Character
        if scene is None:
            self.scene = self.load_scene_with_conf(self.conf)
        else:
            assert isinstance(scene, ODEScene)
            self.scene: ODEScene = scene

        if sim_character is None:
            self.character: ODECharacter = self.scene.character0
            if len(self.scene.characters) > 1:
                for _character in self.scene.characters:
                    if "sim" in _character.name.lower():
                        self.character = _character
                        break

        # remove unused joint names in initial sample window
        # new_sample_win = self.conf["character"]["sample_window"]
        # _jname_set = set(self.character.joint_info.joint_names())
        # new_sample_win = {key: value for key, value in new_sample_win.items() if key in _jname_set}
        # self.conf["character"]["sample_window"] = new_sample_win

        self.to_bvh = CharacterTOBVH(self.character, self.scene.sim_fps)
        self.to_bvh.bvh_hierarchy_no_root()
        self.damped_pd: DampedPDControler = DampedPDControler(self.character)
        self.num_joints = len(self.joints)
        self.joint_info.gen_sample_mask(self.joint_info.joint_names())
        # self.joint_info.gen_sample_mask(set(self.conf["character"]["sample_window"].keys()))

        # load contact info
        # self.handle_contact_conf(self.scene, self.conf)

        self.human36_camera_dict_np = CameraParamBuilder.build(dtype=np.float64)
        self.camera_param: CameraParamNumpy = self.parse_camera_param_base()

        # self.target_2d: Optional[SamconTargetPose2d] = None  # For tracking 2d video

        # for tracking bvh input
        self.target: Optional[SamconTargetPose] = None
        self.tar_set: Optional[SetTargetToCharacter] = None
        self.inv_dyn_target: Optional[TargetPose] = None  # Unused for tracking 2d pos

        worker_conf: Dict[str, float] = self.conf["worker"]
        self.n_sample: Optional[int] = None
        self.n_save: Optional[int] = None

        self.sample_fps: Optional[int] = None
        self.sim_cnt: Optional[int] = None
        self.n_iter: Optional[int] = None
        self.cost_bound: float = worker_conf["cost_bound"]  # The top cost_bound samples is dropped

        self.scene.extract_contact = False  # For faster simu

        # load joint weights from config file
        # _joint_weights = self.character.joint_info.set_joint_weights(self.conf["character"]["joint_weights"])

        # self.num_track_joints = len(self.conf["character"]["sample_window"])
        # if self.num_track_joints < self.num_joints:
        #     track_names = set(self.conf["character"]["sample_window"])
        #     self.track_joint_index: Optional[np.ndarray] = np.array(
        #         [index for index, name in enumerate(self.joint_info.joint_names())
        #         if name in track_names])
        #     with_root_index = np.concatenate([np.array([0]), self.track_joint_index + 1])
        # else:
        #     self.track_joint_index: Optional[np.ndarray] = None
        #     with_root_index = None

        self.loss: SamconLoss = SamconLoss(self.conf)
        self.loss.joint_subset = None
        self.loss.joint_weights = self.joint_info.weights

        self.set_sample_fps(worker_conf["sample_fps"])

        self.use_dense_loss: bool = worker_conf["use_dense_loss"]  # TODO.. implement dense loss
        # however, dense loss seems not work on tracking task..

    @staticmethod
    def get_human36_param(camera_dict: Dict[str, Any], config: Dict[str, Any]):
        return camera_dict[config["name"]][config["index"]]

    def parse_camera_param_base(self):
        """
        load camera parameter for track 2d pose
        """
        camera_conf: Dict[str, Any] = self.conf["camera"]
        camera_type: CameraConfigType = CameraConfigType[camera_conf["type"]]
        if camera_type == CameraConfigType.HUMAN_36:
            self.camera_param: CameraParamNumpy = self.get_human36_param(self.human36_camera_dict_np, camera_conf["HUMAN_36"])
        else:
            raise NotImplementedError

        return self.camera_param

    @staticmethod
    def load_scene_with_conf(conf: Union[str, Dict[str, Any]], scene: Optional[ODEScene] = None, scene_json: Optional[Dict] = None) -> ODEScene:
        if isinstance(conf, str):
            conf: Dict[str, Any] = Helper.conf_loader(conf)

        loader: JsonSceneLoader = JsonSceneLoader(scene)
        add_conf = JsonSceneLoader.AdditionalConfig()
        add_conf.use_angle_limit = True
        scene_conf: Dict[str, Any] = conf["scene"]
        add_conf.simulate_fps = scene_conf["fps"]
        add_conf.render_fps = (add_conf.simulate_fps // 2) if (add_conf.simulate_fps >= 50) else add_conf.simulate_fps
        if scene_json is None:
            world_fname: str = conf["filename"]["world"]
            if world_fname.endswith(".pickle"):
                scene: ODEScene = loader.load_from_pickle_file(world_fname, add_conf)
            elif world_fname.endswith(".json"):
                scene = loader.load_from_file(world_fname, add_conf)
            else:
                raise ValueError
        else:
            scene: ODEScene = loader.load_json(scene_json, add_conf)

        sim_type: SimulationType = SimulationType[scene_conf["sim_type"]]
        scene.set_simulation_type(sim_type)
        if len(scene.characters) == 1 and False:
            SamconWorkerBase.handle_contact_conf(scene, conf)

        return scene

    @staticmethod
    def handle_contact_conf(scene: ODEScene, conf: Dict[str, Any], character: Optional[ODECharacter] = None):
        """
        parse contact info from config file
        """
        contact_dict: Dict[str, Union[int, str]] = conf["character"]["contact"]
        if character is None:
            character = scene.character0
        assert character in scene.characters
        scene.contact_count = contact_dict["count"]
        scene.contact_type = ContactType[contact_dict["type"]]
        character.self_collision = contact_dict["self_collision"]
        character.set_geom_max_friction(contact_dict["contact_max_force_ratio"])
        scene.self_collision = contact_dict["self_collision"]
        soft_contact_dict = contact_dict["soft_contact"]
        if soft_contact_dict["in_use"]:
            scene.use_soft_contact = True
            scene.soft_cfm = soft_contact_dict["SoftCFM"]
            scene.soft_erp = soft_contact_dict["SoftERP"]
        else:
            scene.use_soft_contact = False
        SamconWorkerBase.set_character_enable_contact(character, contact_dict["no_falldown_contact"])
        scene.space.ResortGeoms()

    def set_sample_fps(self, value: int):
        """
        set sample frequency. the default value is 0.1 s (in original samcon paper)
        """
        self.sample_fps = value
        self.sim_cnt: int = int(self.scene.sim_fps / self.sample_fps)

    def init_n_sample(self):
        """
        initial sample count
        """
        if self.worker_size > 1:
            self.n_sample = (self.n_sample // self.worker_size + 1) * self.worker_size

    def calc_n_iter(self) -> int:
        self.n_iter: int = int(self.num_frames / self.fps * self.sample_fps)
        return self.n_iter

    @staticmethod
    def set_character_enable_contact(character: ODECharacter, enable_body_list: Optional[List[str]]):
        character.disable_all_clung_env()
        if enable_body_list is not None:
            character.set_clung_env(enable_body_list, True)

    @property
    def worker_size(self) -> int:
        """
        size of workers
        """
        return self.worker_info.worker_size

    @property
    def comm(self) -> MPI.Intracomm:
        return MPI.COMM_WORLD

    @property
    def comm_rank(self) -> int:
        """
        mpi rank
        """
        return self.worker_info.comm_rank

    @property
    def comm_size(self) -> int:
        """
        mpi size
        """
        return self.worker_info.comm_size

    @property
    def node_name(self) -> str:
        """
        mpi node name
        """
        return self.worker_info.node_name

    @property
    def num_frames(self) -> int:
        return self.target.num_frames

    @property
    def fps(self):
        return self.scene.sim_fps
