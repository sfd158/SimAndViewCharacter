import enum
import json
import os

from matplotlib.pyplot import sca

fdir = os.path.dirname(__file__)
input_fname = os.path.join(fdir, "world-leg3.json")
output_fname = os.path.join(fdir, "world-leg3-long.json")

with open(input_fname, "r") as fin:
    input_res = json.load(fin)

upper_offset = 0.5

character = input_res["CharacterList"]["Characters"][0]
body_list = character["Bodies"]
joint_list = character["Joints"]
end_list = character["EndJoints"]

body_names = [node["Name"] for node in body_list]
body_map = {node: idx for idx, node in enumerate(body_names)}
lower_body_name = [node for node in body_names if "Leg" in node or "Foot" in node or "Toe" in node]
upper_body_name = list(set(body_names) - set(lower_body_name))
# print(lower_body_name)
# print(upper_body_name)

joint_names = [node["Name"] for node in joint_list]
joint_map = {node: idx for idx, node in enumerate(joint_names)}
lower_joint_names = [node for node in joint_names if "Leg" in node or "Ankle" in node or "Toe" in node]
upper_joint_names = list(set(joint_names) - set(lower_joint_names))
# print(lower_joint_names)
# print(upper_joint_names)

# here we should also modify the lower part
for lr in ["r", "l"]:
    leg0 = body_map[f"{lr}Leg0"]
    leg1 = body_map[f"{lr}Leg1"]
    leg2 = body_map[f"{lr}Leg2"]
    joint1 = joint_map[f"{lr}LegJoint1"]
    joint2 = joint_map[f"{lr}LegJoint2"]
    ankle = joint_map[f"{lr}Ankle"]
    hip = joint_map[f"{lr}Hip"]
    ori_hip, ori_ankle = joint_list[hip]["Position"][1], joint_list[ankle]["Position"][1]
    old_len = ori_hip - ori_ankle
    new_len = old_len + upper_offset
    ratio = new_len / old_len
    for node in [leg0, leg1, leg2]:
        info = body_list[node]
        info["Position"][1] = ratio * (info["Position"][1] - ori_ankle) + ori_ankle
        geom = body_list[node]["Geoms"][0]
        geom["Position"][1] = info["Position"][1]
        scale = geom["Scale"]
        scale[1] = ratio * (scale[1] + 2 * scale[0]) - 2 * scale[0]
    for node in [joint1, joint2]:
        info = joint_list[node]
        info["Position"][1] = ratio * (info["Position"][1] - ori_ankle) + ori_ankle

print(upper_body_name)
for node in upper_body_name:
    info = body_list[body_map[node]]
    info["Position"][1] += upper_offset
    for geom in info["Geoms"]:
        geom["Position"][1] += upper_offset

for node in upper_joint_names:
    info = joint_list[joint_map[node]]
    info["Position"][1] += upper_offset

end_name = [node["Name"] for node in end_list]
end_map = {node: idx for idx, node in enumerate(end_name)}
for node in ["rHand", "lHand", "head"]:
    info = end_list[end_map[node]]
    info["Position"][1] += upper_offset

with open(output_fname, "w") as fout:
    json.dump(input_res, fout)
