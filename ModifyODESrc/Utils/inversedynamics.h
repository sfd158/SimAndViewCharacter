#pragma once
#include <Eigen/Dense>
#include <EigenExtension/Mat3Euler.h>
#include <EigenExtension/EigenBindingWrapper.h>
#include <vector>
#include <string>


class CInverseDynamics
{
public:
    CInverseDynamics(); //  (bool flagUseRealInertia /*= false*/);

    CInverseDynamics(
        const std::vector<double> & body_mass, 
        const std::vector<Matrix3d> & body_inertia, 
        const std::vector<Vector3d> & body_position,
        const std::vector<Matrix3d> & body_rotation,
        const std::vector<int> & parent_joint_dof,
        const std::vector<Vector3d> & parent_joint_pos,
        const std::vector<std::string> & parent_joint_euler_order,
        const std::vector<Matrix3d> & parent_joint_euler_axis,
        const std::vector<int> & parent_body_index);

    ~CInverseDynamics(void);

protected:
    void initialize_resize(size_t body_count);
    void set_euler_callback(int i, int jointType);

public:
    // return the dof
    int GeneralizedCoordinatesDimension(void) const { return (int)m_q.size(); }

    // convert root position and joint rotations to generalized coordinates

    void ConvertToGeneralizedCoordinates(
        const Vector3d& rootPos, const std::vector<Quaterniond>& vJointRots,
        std::vector<double>& q) const;

    void ConvertToGeneralizedCoordinates(
        int bodyId, const Quaterniond& jointRot,
        std::vector<double>& q) const;

    Quaterniond ConvertToJointRotation(
        const std::vector<double>& q, int bodyId
    ) const;

    void ConvertToGeneralizedCoordinates(
        int bodyId, const Quaterniond& jointRot,
        double* q) const;

    Quaterniond ConvertToJointRotation(
        const double* q, int bodyId
    ) const;

    void ConvertToJointRotations(
        const std::vector<double>& q,
        Vector3d& rootPos, std::vector<Quaterniond>& vJointRots
    ) const;

    void InitializeInverseDynamics(
        const std::vector<double>& q, const std::vector<double>& qdot,
        const std::vector<double>& qdotdot
    );

    // forward iteration
    void ComputeVelocityAcceleration(const Vector3d& gravity);

    // backword iteration
    void ComputeForceTorque(
        const std::vector<Vector3d>& fext = std::vector<Vector3d>(),
        const std::vector<Vector3d>& text = std::vector<Vector3d>()
    );

    // velocities in body's local frame
    const std::vector<Vector3d>& GetLocalLinearVelocity(void) const { return m_localLinVels; }
    const std::vector<Vector3d>& GetLocalAngularVelocity(void) const { return m_localAngVels; }
    const std::vector<Vector3d>& GetLocalLinearAcceleration(void) const { return m_localLinAccels; }
    const std::vector<Vector3d>& GetLocalAngularAcceleration(void) const { return m_localAngAccels; }

    // torques in body's local frame
    const std::vector<Vector3d>& GetLocalForce(void) const { return m_localForces; }
    const std::vector<Vector3d>& GetLocalTorque(void) const { return m_localTorques; }

    // rotation
    const std::vector<Matrix3d>& GetJointRotation(void) const { return m_R; }
    const std::vector<Matrix3d>& GetBodyOrientation(void) const { return m_R0; }

    // joint dof
    const std::vector<int>& GetJointQIndex(void) const { return m_jointQIndex; }
    size_t GetTotalJointDof() const {
        size_t res = 3;
        for (size_t i = 0; i < m_jointAxies.size(); i++)
        {
            res += m_jointAxies[i].size();
        }
        return res;
    }
    int GetJointDof(int bid) const { return (int)m_jointAxies[bid].size(); }

    // re-root the character to some body, the new root joint will be located at the CoM of the body
    // this operation will only effect the computation on force and torque
    void ReRoot(int bodyId);
    // re-compute joint rotations in new hierarchy
    void ReComputeR(void);
    void ClearReRootFlag(void) { m_flagReRooted = false; }

    // body positions
    void ComputeBodyPositions(std::vector<Vector3d>& pos) const;

    // momentums
    void ComputeCoMMomentums(
        double& mass, Vector3d& com,
        Vector3d& linVelocity, Vector3d& angMomentum,
        Matrix3d& inertia
    ) const;

    // compute (\partial v_c / \partial q), (\partial v_c / \partial qdot)
    //   (\partial L / \partial q), and (\partial L / \partial qdot)

    void ComputeMomentumRootJacobians(
        std::vector<Vector3d>& Jvq, std::vector<Vector3d>& JLq,
        std::vector<Vector3d>& Jvqdot, std::vector<Vector3d>& JLqdot
    ) const;

    // compute (\partial w_i / \partial q_i)
    void ComputeLocalW_q(std::vector<Vector3d>& Jwq, int bodyId) const;

    // void GetJwJoint() const;
    const std::vector<std::vector<Vector3d>>& GetJwJoint(void) const { return m_JwJoint; }
    const std::vector<std::vector<Vector3d>>& GetJwDotJoint(void) const { return m_JwdotJoint; }

    const std::vector<Vector3d>& GetCls(void) const { return m_cls; } // CoM in local
    const std::vector<Vector3d>& GetDCls(void) const { return m_dcls; } // from child joint to com

    size_t size() const { return this->m_Ms.size(); }

    void print_joint_axies() const {}

protected:
    std::vector<std::vector<Vector3d>> m_jointAxies;
    typedef void (*EulerFactor)(const Matrix3d&, double&, double&, double&);
    typedef Matrix3d(*EulerMaker)(double xAngle, double yAngle, double zAngle);
    std::vector<Matrix3d> m_jointOrientMatrix;
    std::vector<EulerFactor> m_jointEulerFactors;
    std::vector<EulerMaker> m_jointEulerMakers;
    std::vector<Vector3d> m_cls; // child body CoM in joint local coordinate
    std::vector<Vector3d> m_dcls; // from child joint to com
    std::vector<Matrix3d> m_Is; // inertia in local frame
    std::vector<double> m_Ms; // masses

    std::vector<std::vector<Vector3d>> m_JwJoint;
    std::vector<std::vector<Vector3d>> m_JwdotJoint;
    std::vector<Matrix3d> m_R;
    std::vector<Matrix3d> m_R0;

    std::vector<Vector3d> m_localLinVels;
    std::vector<Vector3d> m_localAngVels;
    std::vector<Vector3d> m_localLinAccels;
    std::vector<Vector3d> m_localAngAccels;

    std::vector<Vector3d> m_localForces;
    std::vector<Vector3d> m_localTorques;

    std::vector<double> m_q;
    std::vector<double> m_qdot;
    std::vector<double> m_qdotdot;

    std::vector<int> m_jointQIndex;
    std::vector<int> m_order;
    std::vector<int> m_parent;
    std::vector<std::string> parent_joint_euler;

    ////////////////////////
    // for force/torque computation
    // we will do the backward iteration of RENA for the re-rooted hierarchy
    std::vector<int> m_reOrder;
    std::vector<int> m_reOrderBackMap;
    std::vector<int> m_reParent;
    std::vector<Vector3d> m_reCls; // CoM in local
    std::vector<Vector3d> m_reDcls; // from child joint to com
    std::vector<Matrix3d> m_reR;
    bool m_flagReRooted;
};

typedef CInverseDynamics* CInverseDynamicsPtr;
