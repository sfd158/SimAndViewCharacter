# Configurations for SMPL Model

# smpl_name_list
# The number of parameters in SMPL model


smpl_rho = 1000
smpl_param_size = 10
smpl_joint_size = 24
smpl_export_body_size = 22
smpl_export_joint_size = 21
smpl_export_endjoint_size = 21
smpl_vertice_size = 6890
smpl_param_range = 2
smpl_render_color = ['#FFC0CB','#6A5ACD','#B0C4DE','#3CBA71','#F0E68C','#FFA07A',
                     '#EE82EE','#E6E6FA','#F0F8FF','#228B22','#FFD700','#FF6347',
                     '#FF00FF','#0000FF','#00FFFF','#9ACD31','#FFA500','#A52A2A',
                     '#8A2BE2','#4169E1','#40E0D0','#FFFF00','#D2B48C','#D2691E']
# smpl_name_list
# The name of each joint in smpl Model
smpl_name_list = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 
                  'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
                  'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
                  'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']
# smpl_parent_list
# The index of parent node of each joint in smpl Model, -1 if non exists
smpl_parent_list = [-1, 0, 0, 0, 1, 2,
                     3, 4, 5, 6, 7, 8,
                     9, 9, 9, 12, 13, 14,
                     16, 17, 18, 19, 20, 21]
#smpl_num_joint
#the number of joints in SMPL Model
smpl_num_joint = len(smpl_name_list)

# smpl_hierarchy 
# Map the name of joint into the index of the joint in SMPL Model.
smpl_hierarchy = {}
for i in range(0, smpl_num_joint):
    smpl_hierarchy[smpl_name_list[i]] = i


# smpl_parent_info 
# The name of parent joint of each joint, '' on non-exist
smpl_parent_info = {}
for i in range(smpl_joint_size):
    if smpl_parent_list[i] != -1:
        smpl_parent_info[smpl_name_list[i]] = smpl_name_list[smpl_parent_list[i]]
        

# smpl_children_list
# The list of index, which is the children of the specific joint
smpl_children_list = [[] for i in range(0, smpl_num_joint)]
for i in range(0, smpl_num_joint):
    if smpl_parent_list[i] != -1:
        smpl_children_list[smpl_parent_list[i]].append(i)

# smpl_render_index
# The index of the part which need to be rendered
"""
smpl_render_index = [smpl_hierarchy['L_Knee'],
                     smpl_hierarchy['R_Knee'],
                     smpl_hierarchy['L_Ankle'],
                     smpl_hierarchy['R_Ankle'],
                     smpl_hierarchy['L_Foot'],
                     smpl_hierarchy['R_Foot']]
"""
# If you want to render all parts, use the following line
smpl_render_index = [i for i in range(smpl_joint_size)]

# Define the addition point used in tetgen
smpl_tetgen_index = [3, 6, 9,              # Spine       
                     4, 5,                 # Left/Right Leg
                     18, 20, 19, 21]       # Left/Right Arm
smpl_tetgen_radius = [0.06, 0.06, 0.06,       # Spine
                      0.03, 0.03,             # Left/Right Leg
                      0.02, 0.01, 0.02, 0.01] # Left/Right arm
# configuration for export json 
WorldAttr = {
    "FixedAttr": {
        "SimulateFPS": 120,
        "UseHinge": True,
        "UseAngleLimit": False,
        "SelfCollision": False,
        "dWorldUpdateMode": 1
    },
    "ChangeAttr": {
        "Gravity": [
            0.0,
            -9.8,
            0.0
        ],
        "StepCount": 0,
        "RenderFPS": 60
    }
}

Environment = {
    "Geoms": [
        {
            "GeomID": 18990,
            "Name": "Plane",
            "GeomType": "Cube",
            "Collidable": True,
            "Friction": 0.8,
            "Restitution": 1.0,
            "ClungEnv": False,
            "Position": [
                0.0,
                -0.5,
                0.0
            ],
            "Quaternion": [
                0.0,
                0.0,
                0.0,
                1.0
            ],
            "Scale": [
                200.0,
                1.0,
                200.0
            ]
        }
    ],
    "FloorGeomID": 0
}

basicCharacterInfo = {
    "CharacterID": 0,
    "CharacterName": "DCharacter0",
    "SelfCollision": True,
    "IgnoreParentCollision": True,
    "IgnoreGrandpaCollision": True,
    "Kinematic": False,
    "CharacterLabel": "",
    "HasRealRootJoint": False
}


PDControlParam = {
    "Kps": [400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 
            400.0, 400.0, 400.0, 10.0 , 10.0 , 400.0,
            400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
            400.0, 5.0,   5.0],
    "TorqueLimit": [400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 
                    400.0, 400.0, 400.0, 10.0 , 10.0 , 400.0,
                    400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                    400.0, 10.0  ,10.0]
}
