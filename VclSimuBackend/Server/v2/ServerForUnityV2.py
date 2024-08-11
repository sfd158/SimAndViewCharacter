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
import json
import threading
from typing import Any, Dict, Optional, List

from .ODEToUnity import ODEToUnity, UnityDWorldUpdateMode
from .RemoveParse import RemoveParse
from ..v1.ServerBase import ServerBase
from ...ODESim.Loader.JsonSceneLoader import JsonSceneLoader
from ...ODESim.ODEScene import ODEScene
from ...ODESim.ODECharacter import ODECharacter
from ...ODESim.UpdateSceneBase import UpdateSceneBase, SimpleUpdateScene, SimpleDampedUpdateScene


class ServerThreadHandle:
    """
    for visualize in Unity
    run server as main thread, simulation as sub-thread.
    after one step, the forward simulation thread should be hang up for visualizing in Unity..

    implement by threading.Event
    """
    sub_thread_flag = threading.Event()
    end_trial_event = threading.Event()

    @classmethod
    def pause_sub_thread(cls):
        cls.sub_thread_flag.clear()

    @classmethod
    def resume_sub_thread(cls):
        cls.end_trial_event.set()
        cls.sub_thread_flag.set()

    @classmethod
    def sub_thread_wait_for_run(cls):
        cls.sub_thread_flag.wait()
        cls.sub_thread_flag.clear()

    @classmethod
    def sub_thread_run_end(cls):
        cls.end_trial_event.clear()

    @classmethod
    def sub_thread_is_running(cls):
        return cls.end_trial_event.is_set()

    @classmethod
    def wait_sub_thread_run_end(cls):
        cls.end_trial_event.wait()


class ServerMessageType(enum.IntEnum):  # Same as client part
    SUCCESS = 0
    FAIL = 1
    HIERARCHY = 2
    UPDATE = 3
    INITIAL_INSTRUCTION = 4


class ServerForUnityV2(ServerBase):
    """
    Server for rendering in Unity
    """
    sim_str = "Simulation"
    gt_str = "Truth"

    def __init__(
        self,
        scene: Optional[ODEScene] = None,
        update_scene: Optional[UpdateSceneBase] = None,
        ip_addr: str = "localhost",
        ip_port: int = 8888
    ):
        super(ServerForUnityV2, self).__init__(ip_addr, ip_port)
        if scene is None:
            scene = ODEScene()
        self.scene: ODEScene = scene
        if update_scene is None:
            update_scene: Optional[UpdateSceneBase] = SimpleUpdateScene(self.scene)
        self.update_scene: UpdateSceneBase = update_scene
        self.ode_to_unity: ODEToUnity = ODEToUnity(self.scene)

        self.remove_parser: RemoveParse = RemoveParse(self.scene)

        self.print_mess: bool = False
        self.update_export_info: bool = True

        self.init_instruction_buf: Optional[Dict] = None

        self.load_scene_hook = None

    def reset(self):  # reset scene
        self.scene.clear()
        return self

    @staticmethod
    def recieve_success() -> Dict[str, int]:
        return {"MessType": ServerMessageType.SUCCESS.value}

    @staticmethod
    def recieve_fail() -> Dict[str, int]:
        return {"MessType": ServerMessageType.FAIL.value}

    def after_load_hierarchy(self):  # default callback after load hierarchy from Unity client.
        pass

    def select_sim_ref(self):
        """
        When there are 2 characters (1 for simulation, 1 for reference motion)
        divide those characters.
        """
        select_dict = {character.name: idx for idx, character in enumerate(self.scene.characters)}
        if self.init_instruction_buf:
            sim_character: ODECharacter = self.scene.characters[select_dict[self.sim_str]]
            ref_character: ODECharacter = self.scene.characters[select_dict[self.gt_str]]
            ref_character.set_ode_space(None)
            ref_character.is_enable = False
        else:
            sim_character: ODECharacter = self.scene.character0
            ref_character: Optional[ODECharacter] = None

        return sim_character, ref_character

    def handle_control_signal(self, world_signal: Dict[str, Any]):
        if world_signal is None:
            return None
        character_signals: List[Dict[str, Any]] = world_signal.get("CharacterSignals")
        if character_signals is None:
            return None
        # becasue some characters may be removed, so we should remove them..
        character_dict: Dict[int, int] = {character.character_id: ch_idx for ch_idx, character in enumerate(self.scene.characters)}
        ret_dict = {character_dict[node["CharacterID"]]: node
            for node in character_signals if node["CharacterID"] in character_dict}

        # re-order the control signals
        ret_list = [ret_dict[ch_idx] for ch_idx in range(len(self.scene.characters))]
        for ch_idx, character in enumerate(self.scene.characters):
            assert ret_list[ch_idx]["CharacterID"] == character.character_id

        return ret_list

    def calc(self, mess_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        handle request from Unity Client
        """
        
        if self.print_mess:
            print(json.dumps(mess_dict))
        mess_type = mess_dict["MessType"]
        if mess_type == ServerMessageType.HIERARCHY:  # load hierarchy info from Unity
            if self.load_scene_hook is None:
                loader = JsonSceneLoader(self.scene)
                self.scene = loader.load_json(mess_dict["WorldInfo"])
            else:
                self.scene = self.load_scene_hook(self.scene, mess_dict["WorldInfo"])
            fixed_attr = mess_dict["WorldInfo"]["WorldAttr"].get("FixedAttr")
            if fixed_attr and "dWorldUpdateMode" in fixed_attr:
                self.ode_to_unity.unity_update_mode = UnityDWorldUpdateMode(fixed_attr["dWorldUpdateMode"])

            print("Load Scene from Unity Client.")
            result: Dict[str, int] = self.recieve_success()
            self.after_load_hierarchy()
        elif mess_type == ServerMessageType.INITIAL_INSTRUCTION:
            # here we should add new character in python...
            result = self.recieve_success()
            if self.init_instruction_buf:
                result.update(self.init_instruction_buf)
        elif mess_type == ServerMessageType.UPDATE:
            # parse remove info
            remove_info: Optional[Dict[str, Any]] = mess_dict.get("RemoveInfo")
            if remove_info:
                self.remove_parser.parse(remove_info)

            # get control signal. we can control the character via world signal
            if self.update_scene is not None:
                self.update_scene.world_signal = self.handle_control_signal(mess_dict.get("WorldControlSignal"))

            # parse update attrs
            export_info: Optional[Dict[str, Any]] = mess_dict.get("ExportInfo")
            if export_info and self.update_export_info:  # create new character
                loader: JsonSceneLoader = JsonSceneLoader(self.scene, True)
                loader.load_json(export_info)
            
            not_update = False
            if self.update_scene is not None:
                # print(len(self.scene.characters), len(self.update_scene.scene.characters))
                not_update = self.update_scene.update(mess_dict)
            if not_update:
                result = {}
            else:
                result = {"WorldUpdateInfo": self.ode_to_unity.to_dict()}

        else:
            result = self.recieve_fail()
        if self.print_mess:  # for debug..
            print(json.dumps(result))
        return result


def main():
    """
    Simple test case for server
    """
    scene = ODEScene()
    server = ServerForUnityV2(scene, SimpleUpdateScene(scene))
    server.run()


if __name__ == "__main__":
    main()
