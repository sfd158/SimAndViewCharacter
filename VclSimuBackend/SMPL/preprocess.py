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
import pickle
import sys

output_path = './model.pkl'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: expected source model path.')
        exit(-1)
    src_path = sys.argv[1]
    with open(src_path, 'rb') as f:
        src_data = pickle.load(f, encoding="latin1")
    model = {
        'J_regressor': src_data['J_regressor'],
        'weights': np.array(src_data['weights']),
        'posedirs': np.array(src_data['posedirs']),
        'v_template': np.array(src_data['v_template']),
        'shapedirs': np.array(src_data['shapedirs']),
        'f': np.array(src_data['f']),
        'kintree_table': src_data['kintree_table']
    }
    if 'cocoplus_regressor' in src_data.keys():
        model['joint_regressor'] = src_data['cocoplus_regressor']
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
