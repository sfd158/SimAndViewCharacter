import enum
import numpy as np
from typing import Optional, List, Dict
import json


class PrimitiveType(enum.Enum):
    """
    PrimitiveType in Unity.
    """
    Sphere = 0
    Capsule = 1
    Cylinder = 2
    Cube = 3
    Plane = 4


class UnityInfoBaseType:
    def __init__(self, precision: int = 8):
        self.create_id: Optional[np.ndarray] = None
        self.create_type: Optional[np.ndarray] = None
        self.create_scale: Optional[np.ndarray] = None
        self.create_pos: Optional[np.ndarray] = None
        self.create_quat: Optional[np.ndarray] = None
        self.create_child_quat: Optional[np.ndarray] = None
        # numpy string type is not easy to use..
        self.create_name: Optional[List[str]] = None
        self.create_color: Optional[np.ndarray] = None

        self.modify_id: Optional[np.ndarray] = None
        self.modify_pos: Optional[np.ndarray] = None
        self.modify_quat: Optional[np.ndarray] = None

        self.remove_id: Optional[np.ndarray] = None

        self.precision = precision

    def __len__(self):
        if self.create_id is None:
            return 0
        else:
            return self.create_id.size

    """
    def ndarray_to_str(self, a: Optional[np.ndarray]):
        return np.array2string(a, max_line_width=sys.maxsize,
                                   precision=self.precision if a.dtype == np.float64 else None,
                                   separator=",", threshold=sys.maxsize).replace(" ", "")
    """

    def gen_ran_color(self):
        self.create_color = np.random.uniform(0.3, 1, len(self) * 3)

    def calc_init_name(self):
        self.create_name = [PrimitiveType(i).name for i in self.create_type]

    def clear(self):
        self.create_id: Optional[np.ndarray] = np.array([])
        self.create_type: Optional[np.ndarray] = np.array([])
        self.create_scale: Optional[np.ndarray] = np.array([])
        self.create_pos = np.array([])
        self.create_quat = np.array([])
        self.create_child_quat = np.array([])
        self.create_name = []
        self.create_color = np.array([])

        self.modify_id = np.array([])
        self.modify_pos = np.array([])
        self.modify_quat = np.array([])

        self.remove_id = np.array([])

    def to_json_dict(self) -> Dict[str, List]:
        return {
            "CreateID": self.create_id.tolist(),
            "CreateType": self.create_type.tolist(),
            "CreateScale": self.create_scale.tolist(),
            "CreatePos": self.create_pos.tolist(),
            "CreateQuat": self.create_quat.tolist(),
            "CreateChildQuat": self.create_child_quat.tolist(),
            "CreateName": self.create_name,
            "CreateColor": self.create_color.tolist(),
            "ModifyID": self.modify_id.tolist(),
            "ModifyPos": self.modify_pos.tolist(),
            "ModifyQuat": self.modify_quat.tolist(),
            "RemoveID": self.remove_id.tolist()
        }

    def to_json(self):
        return json.dumps(self.to_json_dict())
