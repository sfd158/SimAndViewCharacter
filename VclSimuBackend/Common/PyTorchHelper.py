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

import numpy as np
import torch
from typing import Any, Dict, List

cpu_device = torch.device("cpu")


class ConvertTorch:
    """
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def handle_case(value, dtype=torch.float32, device=torch.device("cpu")):
        if isinstance(value, List):
            result = ConvertTorch.handle_list(value, dtype, device)
        elif isinstance(value, Dict):
            result = ConvertTorch.handle_dict(value, dtype, device)
        elif isinstance(value, np.ndarray):
            result = torch.as_tensor(value, dtype=dtype, device=device)
        elif isinstance(value, torch.Tensor):
            result = torch.as_tensor(value.detach(), dtype=dtype, device=device)
        else:
            raise ValueError
        return result

    @staticmethod
    def handle_dict(mess_dict: Dict[str, Any], dtype=torch.float32, device=torch.device("cpu")) -> Dict[str, Any]:
        result = {}
        for key, value in mess_dict.items():
            result[key] = ConvertTorch.handle_case(value, dtype, device)
        return result

    @staticmethod
    def handle_list(mess_list: List, dtype=torch.float32, device=torch.device("cpu")):
        result = []
        for value in mess_list:
            result.append(ConvertTorch.handle_case(value, dtype, device))
        return result


class ConvertNumpy:
    def __init__(self) -> None:
        pass

    @staticmethod
    def handle_case(value, dtype=np.float64):
        if isinstance(value, List):
            result = ConvertNumpy.handle_list(value, dtype)
        elif isinstance(value, Dict):
            result = ConvertNumpy.handle_dict(value, dtype)
        elif isinstance(value, np.ndarray):
            result = value.astype(dtype)
        elif isinstance(value, torch.Tensor):
            result = value.detach().cpu().clone().numpy().astype(dtype)
        else:
            raise ValueError
        return result

    @staticmethod
    def handle_dict(mess_dict: Dict[str, Any], dtype=np.float64) -> Dict[str, Any]:
        result = {}
        for key, value in mess_dict:
            result[key] = ConvertNumpy.handle_case(value, dtype)
        return result

    @staticmethod
    def handle_list(mess_list: List, dtype=np.float64):
        result = []
        for value in mess_list:
            result.append(ConvertNumpy.handle_case(value, dtype))
        return result


class PyTorchHelper:
    def __init__(self) -> None:
        pass

    @staticmethod
    def dict_from_numpy(mess_dict: Dict[str, np.ndarray], dtype=torch.float32, device=cpu_device):
        result = {key: torch.as_tensor(value, dtype=dtype, device=device) for key, value in mess_dict.items()}
        return result

    @staticmethod
    def dict_to_numpy(mess_dict: Dict[str, torch.Tensor], dtype=np.float32):
        result = {key: value.detach().cpu().clone().numpy().astype(dtype) for key, value in mess_dict.items()}
        return result


