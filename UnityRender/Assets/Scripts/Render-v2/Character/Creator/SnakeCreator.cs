/*
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
*/
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace RenderV2
{
    public class SnakeCreator : MonoBehaviour
    {
        public Vector3 Center;
        public float BallRadius;
        public int NumJoint;
        public DJointType JointType;

        public float Kp = 10;
        public float Kd = 5;
        public float MaxTorque = 40;

        protected DCharacterList dCharacterList;

        public DCharacter CreateFactory()
        {
            if (dCharacterList == null)
            {
                dCharacterList = GetComponent<DCharacterList>();
            }

            GameObject characterObj = new GameObject();
            characterObj.name = "OneHotTree";
            DCharacter dCharacter = characterObj.AddComponent<DCharacter>();

            DRigidBody rootbody = DRigidBody.CreateBallBody(characterObj, 0, "Center", Center, BallRadius);
            DEmptyJoint RootJoint = DEmptyJoint.CreateEmptyJoint(rootbody, characterObj);
            RootJoint.gameObject.name = "RootJoint";
            RootJoint.transform.parent = characterObj.transform;
            dCharacter.transform.position = Center;
            RootJoint.transform.position = Center;
            rootbody.transform.position = Center;

            DJoint lastJoint = RootJoint;
            for(int i = 0; i < NumJoint; i++)
            {
                Vector3 joint_pos = new Vector3(Center.x, Center.y, Center.z + 1.25F * ((i + 1) * BallRadius - 0.5F * BallRadius));
                Vector3 pos = new Vector3(Center.x, Center.y, Center.z + 1.25F * (i + 1) * BallRadius);
                //Debug.Log(pos);
                DRigidBody body = DRigidBody.CreateBallBody(characterObj, 0, "Node" + i, pos, BallRadius);
                
                DJoint joint = null;
                switch (JointType)
                {
                    case DJointType.BallJointType:
                        joint = DBallJoint.AddBallJoint(i, Kd, joint_pos, characterObj);
                        break;
                    case DJointType.HingeJointType:
                        joint = DHingeJoint.AddHingeJoint(i, Kd, joint_pos, "Y", characterObj);
                        break;
                    case DJointType.FixedJointType:
                        joint = DFixedJoint.AddFixedJoint(i, Kd, Center, characterObj);
                        break;
                    default:
                        throw new System.ArgumentException();
                }

                body.IDNum = i;
                joint.IDNum = i;

                body.transform.parent = joint.transform;
                joint.transform.parent = lastJoint.transform;
                joint.transform.position = joint_pos;
                body.transform.position = pos;

                lastJoint = joint;
            }

            var pDController = PDController.PDControllerCreate(NumJoint, characterObj);
            pDController.DefaultKp = Kp;
            pDController.DefaultTorqueLimit = MaxTorque;
            pDController.SetDefaultKp();
            pDController.SetDefaultTorqueLimit();

            dCharacter.transform.parent = dCharacterList.transform;
            dCharacter.SelfCollision = false;
            dCharacter.ReCompute();
            return dCharacter;
        }
    }
}
