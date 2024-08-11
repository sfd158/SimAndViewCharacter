`main.py` entrance

`SamconLoss.py` Calculate Loss of Samcon

`SamconWorkerMain.py` Main Worker of Samcon. MPI is required.

`SamconTargetPose.py` Target Pose in Samcon

`SamconUpdateScene.py` Start a unity server, and drive the character by saved result of Samcon

`SamconWorker.py` Worker of Samcon. MPI is required.

`SamconWorkerBase.py` Base Class of SamconMainWorker and SamconWorker

`StateTree.py` Tree Structure used in Samcon





Run Samcon:

`mpiexec -n NumOfThreads python main.py`

NumOfThreads >= 2

Start Unity server:

`python SamconUpdateScene.py`

