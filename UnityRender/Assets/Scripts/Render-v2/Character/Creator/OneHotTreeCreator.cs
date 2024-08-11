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
using UnityEngine;

namespace RenderV2
{
    public class OneHotTreeCreator : MonoBehaviour
    {
        public Vector3 Center;
        public float BallRadius;
        public int NumBranch;
        public float BranchRadius;
        public float BranchLength;

        public DJointType JointType;

        public float Kp = 10;
        public float Kd = 5;
        public float MaxTorque = 40;
        //public float Angle;

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

            // create the root component
            DRigidBody rootbody = DRigidBody.CreateBallBody(characterObj, 0, "Center", Center, BallRadius);
            DEmptyJoint RootJoint = DEmptyJoint.CreateEmptyJoint(rootbody, characterObj);
            RootJoint.gameObject.name = "RootJoint";
            RootJoint.transform.parent = characterObj.transform;
            dCharacter.transform.position = Center;
            RootJoint.transform.position = Center;
            rootbody.transform.position = Center;

            // Create the child component
            float HalfLen = 0.5F * BranchLength + BranchRadius;
            //float ang = Mathf.Deg2Rad * Angle;
            float ang = 0;
            float y = Center.y - HalfLen * Mathf.Sin(ang);
            float xz = HalfLen * Mathf.Cos(ang);
            
            for(int i=0; i<NumBranch; i++)
            {
                float phi = (2 * i * Mathf.PI) / NumBranch;
                float x = Center.x + (BallRadius + xz) * Mathf.Cos(phi), z = Center.z + (BallRadius + xz) * Mathf.Sin(phi);

                Vector3 pos = new Vector3(x, y, z);
                DRigidBody drigidbody = DRigidBody.CreateCapsuleBody(characterObj, i + 1, "branch" + i, pos, BranchRadius, BranchLength);
                DJoint joint = null;
                switch (JointType)
                {
                    case DJointType.BallJointType:
                        joint = DBallJoint.AddBallJoint(i, Kd, Center, characterObj);
                        break;
                    case DJointType.HingeJointType:
                        joint = DHingeJoint.AddHingeJoint(i, Kd, Center, "Y", characterObj);
                        break;
                    case DJointType.FixedJointType:
                        joint = DFixedJoint.AddFixedJoint(i, Kd, Center, characterObj);
                        break;
                    default:
                        throw new System.ArgumentException();
                }
                
                joint.transform.position = Center;
                drigidbody.transform.position = pos;

                Transform geomtrans = drigidbody.transform.GetChild(0);
                Quaternion quat = Quaternion.FromToRotation(Vector3.forward, pos - Center);
                geomtrans.rotation = quat;

                drigidbody.transform.parent = joint.transform;
                joint.transform.parent = RootJoint.transform;
            }

            // Create PD Controller..
            var pDController = PDController.PDControllerCreate(NumBranch, characterObj);
            pDController.DefaultKp = Kp;
            pDController.DefaultTorqueLimit = MaxTorque;
            pDController.SetDefaultKp();
            pDController.SetDefaultTorqueLimit();

            dCharacter.transform.parent = dCharacterList.transform;
            dCharacter.SelfCollision = false;
            //dCharacter.ReCompute();
            return dCharacter;
        }
    }
}