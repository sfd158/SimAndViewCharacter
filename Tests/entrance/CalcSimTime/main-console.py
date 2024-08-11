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

import time
import sys

from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader, ODEScene


def main():
    loader = JsonSceneLoader()
    fname = "../../CharacterData/world-stdhuman.json"
    # fname = "../../CharacterData/StdHand/world.json"
    scene = loader.load_from_file(fname)

    scene.set_sim_fps(120)
    cnt = 120
    v_time = scene.sim_dt * cnt
    start_collision = time.time()
    for i in range(cnt):
        scene.simulate_once()
    end_collision = time.time()
    use_time = end_collision - start_collision
    print("Step with collision %f, virtual time = %f, ratio = %f" % (use_time, v_time, v_time / use_time))

    scene.reset()
    start_collision = time.time()
    for i in range(cnt):
        scene.step_fast_collision()
    end_collision = time.time()
    use_time = end_collision - start_collision
    print("Step with fast collision %f, virtual time = %f, ratio = %f" % (use_time, v_time, v_time / use_time))

    scene.reset()
    start_collision = time.time()
    for i in range(cnt):
        scene.damped_simulate_once()
    end_collision = time.time()
    use_time = end_collision - start_collision
    print("Damped step with collision %f, virtual time = %f, ratio = %f" % (use_time, v_time, v_time / use_time))

    scene.reset()
    start_collision = time.time()
    for i in range(cnt):
        scene.damped_step_fast_collision()
    end_collision = time.time()
    use_time = end_collision - start_collision
    print("Damped with fast collision %f, virtual time = %f, ratio = %f" % (use_time, v_time, v_time / use_time))


if __name__ == "__main__":
    main()
