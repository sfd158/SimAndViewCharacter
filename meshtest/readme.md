由于mesh的面比较多，可能造成碰撞时的contact数量过大。
在ModifyOde.pyx的collide函数中设置了contact数量的一个上界，目前暂时设置成200；
但为了保证不崩溃，可以设置到2000左右（取决于mesh有多密）

然而宋老师有言：太大是不好的，所以请使用者耗子尾汁

----
2021.11.25

把mesh的画法改为calllist

注意由于ode要求读入的mesh的中心即为重心，并且ode里mesh的mass设置是有问题的，
所以在导入ode前，请修改曲面使其中心与重心重合，并手动设置mass。
具体的操作可以参考`drawObj2.py`。

目前设置最多画10个曲面，可修改`drawstuffWrapper.cpp`的`drawStart()`中的`glGenLists()`的参数来设置。