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
import copy
import json
import numpy as np
import os
from typing import Any, Dict, List, Set

fdir = os.path.dirname(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_fname", type=str, default=os.path.join(fdir, "../../CharacterData/StdHuman/world.json"))
    parser.add_argument("--output_fname", type=str, default="")

    args = parser.parse_args()
    return args


def simple_leg3():
    args = parse_args()
    # load the json data..
    with open(args.input_fname, "r") as f:
        mess_dict: Dict[str, Any] = json.load(f)

    # find the body, joint list..
    character_dict = mess_dict["CharacterList"]["Characters"][0]
    body_list: List[Dict[str, Any]] = character_dict["Bodies"]
    body_names: List[str] = [node["Name"] for node in body_list]
    joint_list = character_dict["Joints"]
    joint_names: List[str] = [node["Name"] for node in joint_list]

    new_joint_list = []
    new_body_list = []

    selected_joint_index = []
    selected_body_index = []

    l_up_leg_idx = body_names.index("lUpperLeg")
    l_low_leg_idx = body_names.index("lLowerLeg")
    r_up_leg_idx = body_names.index("rUpperLeg")
    r_low_leg_idx = body_names.index("rLowerLeg")

    l_hip_jidx = joint_names.index("lHip")
    l_knee_jidx = joint_names.index("lKnee")
    l_ankle_jidx = joint_names.index("lAnkle")

    r_hip_jidx = joint_names.index("rHip")
    r_knee_jidx = joint_names.index("rKnee")
    r_ankle_jidx = joint_names.index("rAnkle")

    base_body_json = copy.deepcopy(body_list[0])
    base_joint_json = copy.deepcopy(joint_list[0])
    # print(base_body_json.keys())
    # print(base_joint_json.keys())

    # compute the middle joint and body
    # here we can change the parent child as string
    def create_new_leg(up_leg_idx_: int, low_leg_idx_: int,
        hip_jidx_: int, knee_jidx_: int, ankle_jidx_: int,
        lr_flag = "l",
        ratio=(0.3, 0.6)
    ):
        selected_joint_index.extend([hip_jidx_, knee_jidx_, ankle_jidx_])
        selected_body_index.extend([up_leg_idx_, low_leg_idx_])

        up_leg_body_ = body_list[up_leg_idx_]
        hip_joint_ = joint_list[hip_jidx_]
        ankle_joint_ = joint_list[ankle_jidx_]
        hip_pos_ = np.array(ankle_joint_["Position"])
        ankle_pos_ = np.array(hip_joint_["Position"])
        jpos1_: np.ndarray = ratio[0] * ankle_pos_ + (1 - ratio[0]) * hip_pos_
        jpos2_: np.ndarray = ratio[1] * ankle_pos_ + (1 - ratio[1]) * hip_pos_
        b_radius_ = up_leg_body_["Geoms"][0]["Scale"][0]
        # create new body json
        new_b0_ = copy.deepcopy(up_leg_body_)
        new_b0_["Name"] = f"{lr_flag}Leg0"
        new_b0_["Position"][1] = 0.5 * (hip_pos_[1] + jpos1_[1])
        new_b0_["Geoms"][0]["Position"][1] = 0.5 * (hip_pos_[1] + jpos1_[1])
        new_b0_["Geoms"][0]["Scale"][1] = (hip_pos_[1] - jpos1_[1]) - 2 * b_radius_
        new_b0_["ParentJointID"] = f"{lr_flag}Hip"
        new_b0_["ParentBodyID"] = f"pelvis"

        new_b1_ = copy.deepcopy(up_leg_body_)
        new_b1_["Name"] = f"{lr_flag}Leg1"
        new_b1_["Position"][1] = 0.5 * (jpos1_[1] + jpos2_[1])
        new_b1_["Geoms"][0]["Position"][1] = 0.5 * (jpos1_[1] + jpos2_[1])
        new_b1_["Geoms"][0]["Scale"][1] = (jpos1_[1] - jpos2_[1]) - 2 * b_radius_
        new_b1_["ParentJointID"] = f"{lr_flag}LegJoint1"
        new_b1_["ParentBodyID"] = f"{lr_flag}"

        new_b2_ = copy.deepcopy(up_leg_body_)
        new_b2_["Name"] = f"{lr_flag}Leg2"
        new_b2_["Position"][1] = 0.5 * (jpos2_[1] + ankle_pos_[1])
        new_b2_["Geoms"][0]["Position"][1] = 0.5 * (jpos2_[1] + ankle_pos_[1])
        new_b2_["Geoms"][0]["Scale"][1] = (jpos2_[1] - jpos2_[1]) - 2 * b_radius_
        new_b2_["ParentJointID"] = f"{lr_flag}LegJoint2"
        new_b2_["ParentBodyID"] = None

        new_j1_ = copy.deepcopy(hip_joint_)
        new_j1_["Name"] = f"{lr_flag}LegJoint1"
        new_j1_["Position"][1] = jpos1_[1]
        new_j1_["ParentJointID"] = f"{lr_flag}Hip"
        new_j1_["ParentBodyID"] = f"{lr_flag}Leg0"
        new_j1_["ChildBodyID"] = f"{lr_flag}Leg1"
        new_j1_["JointType"] = "HingeJoint"
        new_j1_["AngleLoLimit"] = [-180]
        new_j1_["AngleHiLimit"] = [180]
        new_j1_["EulerOrder"] = "X"

        new_j2_ = copy.deepcopy(hip_joint_)
        new_j2_["Name"] = f"{lr_flag}LegJoint2"
        new_j2_["Position"][1] = jpos2_[1]
        new_j2_["ParentJointID"] = f"{lr_flag}LegJoint1"
        new_j2_["ParentBodyID"] = f"{lr_flag}Leg1"
        new_j2_["ChildBodyID"] = f"{lr_flag}Leg2"
        new_j2_["JointType"] = f"HingeJoint"
        new_j2_["JointType"] = "HingeJoint"
        new_j2_["AngleLoLimit"] = [-180]
        new_j2_["AngleHiLimit"] = [180]
        new_j1_["EulerOrder"] = "X"

        print(new_b0_["Position"], "\n")
        print(new_b1_["Position"], "\n")
        print(new_b2_["Position"], "\n")
        print(new_j1_["Position"], "\n")
        print(new_j2_["Position"], "\n")
        print("\n\n\n\n\n==================")
        new_body_list.extend([new_b0_, new_b1_, new_b2_])
        new_joint_list.extend([new_j1_, new_j2_])

    # print(character_dict)
    create_new_leg(l_up_leg_idx, l_low_leg_idx, l_hip_jidx, l_knee_jidx, l_ankle_jidx)
    create_new_leg(r_up_leg_idx, r_low_leg_idx, r_hip_jidx, r_knee_jidx, r_ankle_jidx)

    for i in range(len(joint_list)):
        if joint_list[i]["ParentJointID"] != -1:
            joint_list[i]["ParentJointID"] = joint_list[joint_list[i]["ParentJointID"]]["Name"]
        if joint_list[i]["ParentBodyID"] != -1:
            joint_list[i]["ParentBodyID"] = body_list[joint_list[i]["ParentBodyID"]]["Name"]
        joint_list[i]["ChildBodyID"] = body_list[joint_list[i]["ChildBodyID"]]["Name"]

    for i in range(len(body_list)):
        if body_list[i]["ParentJointID"] != -1:
            body_list[i]["ParentJointID"] = joint_list[body_list[i]["ParentJointID"]]["Name"]
        if body_list[i]["ParentBodyID"] != -1:
            body_list[i]["ParentBodyID"] = body_list[body_list[i]["ParentBodyID"]]["Name"]

    # compute new id
    

if __name__ == "__main__":
    simple_leg3()
