import argparse
import os
from VclSimuBackend.Samcon.SamconCMA.TrajOptDirectBVH import DirectTrajOptBVH, SamconWorkerFull, WorkerInfo, SamHlp, LoadTargetPoseMode
from VclSimuBackend.Render.Renderer import RenderWorld


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_target_mode", type=str,
        choices=LoadTargetPoseMode._member_names_, default="BVH_MOCAP",
        help="target mode. You can track bvh file or inverse dynamics output")
    parser.add_argument("--render", action="store_true", default=True)

    args = parser.parse_args()
    args.load_target_mode = LoadTargetPoseMode[args.load_target_mode]

    info = WorkerInfo()
    samhlp = SamHlp(os.path.join(os.path.dirname(__file__), "../CharacterData/SamconConfig-duplicate.json"))
    worker = SamconWorkerFull(samhlp, info)
    main_worker = DirectTrajOptBVH(samhlp, info, worker, worker.scene, None, args.load_target_mode)

    if args.render:
        render = RenderWorld(worker.scene.world)  # render using Long Ge's framework
        render.start()

    main_worker.test_direct_trajopt()
