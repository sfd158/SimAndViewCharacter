import os
from VclSimuBackend.pymotionlib import BVHLoader
fdir = os.path.dirname(__file__)
fname = os.path.join(fdir, "WalkF-mocap.bvh")
out_fname = os.path.join(fdir, "WalkF-mocap-100.bvh")
mocap = BVHLoader.load(fname)
mocap = mocap.resample(100)
BVHLoader.save(mocap, out_fname)
