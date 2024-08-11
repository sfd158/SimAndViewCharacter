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

from VclSimuBackend.Utils.MothonSliceSmooth import MotionSliceSmooth, BVHLoader, np, MotionData
import subprocess
from scipy.spatial.transform import Rotation


def handle_height(mocap: MotionData):
    # root height is noisy in cmu mocap data. Don't use this function for hand motion.
    min_height = np.min(mocap.joint_position[:, :, 1], axis=1)
    win = 30
    left = win // 2
    right = win - left
    dh = min_height.copy()
    dh[:win] = max(np.min(min_height[:win]), 0.0)
    dh[-win:] = max(np.min(min_height[-win:]), 0.0)
    for i in range(left, mocap.num_frames - right):
        dh[i] = max(np.min(min_height[i-left:i+right]), 0.0)
    mocap.joint_position[:, :, 1] -= dh[:, None]
    mocap.joint_translation[:, :, 1] -= dh[:, None]
    return mocap

# Do not use handle_height function except cmu mocap data..

def handle_S1():
    def S1_sitting_down():
        human36 = BVHLoader.load(r"F:\Human36Reheight\S1\SittingDown-mocap-100-modify.bvh")
        def build_standup_1():  # Kneel down, then get up.
            cmu1 = handle_height(BVHLoader.load(r"F:\cmu-retarget\114\114_02-mocap-100.bvh"))
            getup_motion = cmu1.sub_sequence(750, cmu1.num_frames - 350)
            result = MotionSliceSmooth.append_motion_smooth(human36, getup_motion, 100)
            fname = "SittingDown-standup-1.bvh"
            BVHLoader.save(result, fname)
            subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])

        def build_standup_2():  # lie down, and get up
            cmu_2 = handle_height(BVHLoader.load(r"F:\cmu-retarget\113\113_08-mocap-100.bvh"))
            human36_lie = human36.sub_sequence(3809, 3909)
            cmu_lie_getup = cmu_2.sub_sequence(344, 444)
            lie_down_full = MotionSliceSmooth.merge_motion(human36_lie, cmu_lie_getup)
            human36_start = human36.sub_sequence(0, 802)
            getup_motion = cmu_2.sub_sequence(899, 1532)
            result = MotionSliceSmooth.append_motion_smooth(human36_start, lie_down_full, 30)
            result = MotionSliceSmooth.append_motion_smooth(result, getup_motion, 30)
            fname = "SittingDown-standup-2.bvh"
            BVHLoader.save(result, fname)
            subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])

    human36_fname = r"F:\Human36Reheight\S1\SittingDown2-mocap-100-modify.bvh"
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", human36_fname])
    human36 = BVHLoader.load(human36_fname)
    def build_standup_1():
        cmu1 = handle_height(BVHLoader.load(r"F:\cmu-retarget\114\114_02-mocap-100.bvh"))
        getup_motion = cmu1.sub_sequence(750, cmu1.num_frames - 350)
        getup_motion.joint_translation[:, 0, 1] -= 0.09
        getup_motion.joint_position[:, :, 1] -= 0.09
        result = MotionSliceSmooth.append_motion_smooth(human36, getup_motion, 100)
        fname = "SittingDown2-standup-1.bvh"
        BVHLoader.save(result, fname)
        subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])

    def build_standup_2():
        cmu_2 = handle_height(BVHLoader.load(r"F:\cmu-retarget\113\113_08-mocap-100.bvh"))
        human36_lie = human36.sub_sequence(2178, 2178 + 100)
        cmu_lie_getup = cmu_2.sub_sequence(344, 444)
        lie_down_full = MotionSliceSmooth.merge_motion(human36_lie, cmu_lie_getup)

        human36_start = human36.sub_sequence(0, 2178 + 50)
        getup_motion = cmu_2.sub_sequence(899, 1532)
        result = MotionSliceSmooth.append_motion_smooth(human36_start, lie_down_full, 30)
        result = MotionSliceSmooth.append_motion_smooth(result, getup_motion, 30)
        fname = "SittingDown2-standup-2.bvh"
        BVHLoader.save(result, fname)
        subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])

    # build_standup_1()
    build_standup_2()


def handle_S5():
    def build0():
        human36_fname = r"F:\Human36Reheight\S5\SittingDown-mocap-100.bvh"
        # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", human36_fname])
        human36 = BVHLoader.load(human36_fname)
        def build_standup_1():  # Kneel down, then get up.
            cmu1 = handle_height(BVHLoader.load(r"F:\cmu-retarget\114\114_02-mocap-100.bvh"))
            getup_motion = cmu1.sub_sequence(750, cmu1.num_frames - 350)
            sub_h36m = human36.sub_sequence(0, 5115)
            result = MotionSliceSmooth.append_motion_smooth(sub_h36m, getup_motion, 100)
            fname = "S5-SittingDown-standup-1.bvh"
            BVHLoader.save(result, fname)
            subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])
        
        def build_standup_2():
            cmu2_fname = r"F:\cmu-retarget\113\113_08-mocap-100.bvh"
            cmu_2 = handle_height(BVHLoader.load(cmu2_fname))
            result = MotionSliceSmooth.append_motion_smooth(human36.sub_sequence(5050, None), cmu_2.sub_sequence(1166, None), 100)
            fname = "S5-SittingDown-standup-2.bvh"
            BVHLoader.save(result, fname)
            subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])

    human36_fname = r"F:\Human36Reheight\S5\SittingDown1-mocap-100.bvh"
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", human36_fname])
    human36 = handle_height(BVHLoader.load(human36_fname))
    def build_standup_3():
        # here we need to concat the motion with standing up
        cmu2 = handle_height(BVHLoader.load(r"F:\cmu-retarget\140\140_01-mocap-100.bvh").sub_sequence(1, None))
        result = MotionSliceSmooth.append_motion_smooth(human36.sub_sequence(6468, 7467), cmu2, 100)
        fname = "S5-SittingDown1-standup-1.bvh"
        BVHLoader.save(result, fname)
        subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])

    def build_standup_4():
        cmu = handle_height(BVHLoader.load(r"F:\cmu-retarget\140\140_01-mocap-100.bvh").sub_sequence(376, None))
        result = MotionSliceSmooth.append_motion_smooth(human36, cmu, 10)
        standup = result.sub_sequence(7840, 7877).resample(result.fps * 4)
        standup._fps = result.fps
        mo_a = result.sub_sequence(0, 7840)
        mo_a.append(standup)
        mo_a.append(result.sub_sequence(7877, None))
        fname = "S5-SittingDown1-standup-2.bvh"
        BVHLoader.save(mo_a, fname)
        subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])
    # build_standup_1()
    # build_standup_3()
    build_standup_4()


def handle_S6_0():
    human36_fname = r"F:\Human36Reheight\S6\SittingDown-mocap-100.bvh"
    human36 = BVHLoader.load(human36_fname)
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", human36_fname])
    cmu2_fname = r"F:\cmu-retarget\113\113_08-mocap-100.bvh"
    cmu_2 = handle_height(BVHLoader.load(cmu2_fname))
    result = MotionSliceSmooth.append_motion_smooth(human36, cmu_2.sub_sequence(1166, None), 100)
    # hack: half neck rotation
    print(result.joint_names)
    jindex = result.joint_names.index("torso_head")
    result.joint_rotation[-333:, jindex, :] = Rotation.from_rotvec(0.5 * Rotation(result.joint_rotation[-333:, jindex, :]).as_rotvec()).as_quat()
    fname = "S6-SittingDown-standup.bvh"
    BVHLoader.save(result, fname)
    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])


def handle_S6_1():
    human36_fname = r"F:\Human36Reheight\S6\SittingDown1-mocap-100.bvh"
    human36 = BVHLoader.load(human36_fname)
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", human36_fname])
    cmu2_fname = r"F:\cmu-retarget\113\113_08-mocap-100.bvh"
    cmu_2 = handle_height(BVHLoader.load(cmu2_fname))
    result = MotionSliceSmooth.append_motion_smooth(human36, cmu_2.sub_sequence(1166, None), 100)
    # hack: half neck rotation
    print(result.joint_names)
    jindex = result.joint_names.index("torso_head")
    result.joint_rotation[-333:, jindex, :] = Rotation.from_rotvec(0.5 * Rotation(result.joint_rotation[-333:, jindex, :]).as_rotvec()).as_quat()
    fname = "S6-SittingDown1-standup.bvh"
    BVHLoader.save(result, fname)
    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])


def handle_S7_1():
    human36_fname = r"F:\Human36Reheight\S8\SittingDown1-mocap-100.bvh"
    human36 = BVHLoader.load(human36_fname)
    # subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", human36_fname])
    cmu2_fname = r"F:\cmu-retarget\113\113_08-mocap-100.bvh"
    cmu_2 = handle_height(BVHLoader.load(cmu2_fname))
    result = MotionSliceSmooth.append_motion_smooth(human36, cmu_2.sub_sequence(1166, None), 100)
    # hack: half neck rotation
    print(result.joint_names)
    jindex = result.joint_names.index("torso_head")
    result.joint_rotation[-333:, jindex, :] = Rotation.from_rotvec(0.5 * Rotation(result.joint_rotation[-333:, jindex, :]).as_rotvec()).as_quat()
    fname = "S8-SittingDown1-standup.bvh"
    BVHLoader.save(result, fname)
    subprocess.call(["python", "-m", "VclSimuBackend.pymotionlib.editor", "--bvh_fname", fname])

if __name__ == "__main__":
    handle_S7_1()
