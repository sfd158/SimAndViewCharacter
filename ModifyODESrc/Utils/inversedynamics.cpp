#include <inversedynamics.h>
#include <inertia.h>
#include <iostream>
#include <cctype>

#define InvDynDbg 0
#if InvDynDbg
#define DbgCall std::cout <<  __func__ << "line:" << __LINE__ << std::endl;
#define DbgCallBegin std::cout << "call" << __func__ << "line:" << __LINE__ << std::endl;
#define DbgCallAfter std::cout << "after" << __func__ << "  line:" << __LINE__ << std::endl;
#else
#define DbgCallBegin NULL;
#endif

static int joint_euler_cstr_to_joint_type(const char* s, int jdof)
{
    int jointType = 0;
    for (int j = 0; j < jdof; j++)
    {
        int eid = s[j] - 'X' + 1;
        for (int e = jdof - 1; e > j; --e)
            eid *= 10;
        jointType += eid;
    }
    return jointType;
}

static int joint_euler_str_to_joint_type(const std::string& s)
{
    int jdof = (int)s.size();
    return joint_euler_cstr_to_joint_type(s.c_str(), jdof);
}

static void print_std_vector_Matrix3d(const std::vector<Matrix3d>& a)
{
    for (int i = 0; i < a.size(); i++)
    {
        std::cout << i << ":" << std::endl;
        std::cout << a[i] << std::endl;
    }
    std::cout << std::endl << std::endl;
}

static void print_std_vector_std_string(const std::vector<std::string>& a)
{
    for (int i = 0; i < a.size(); i++)
    {
        std::cout << a[i] << std::endl;
    }
    std::cout << std::endl << std::endl;
}

static void print_std_vector_int(const std::vector<int>& a)
{
    for (int i = 0; i < a.size(); i++)
    {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

CInverseDynamics::CInverseDynamics()
{
    m_flagReRooted = false;
}

CInverseDynamics::CInverseDynamics(
    const std::vector<double> & body_mass, 
    const std::vector<Matrix3d> & body_inertia, 
    const std::vector<Vector3d> & body_position,
    const std::vector<Matrix3d> & body_rotation,
    const std::vector<int> & parent_joint_dof,
    const std::vector<Vector3d> & parent_joint_pos,
    const std::vector<std::string> & parent_joint_euler_order,
    const std::vector<Matrix3d> & parent_joint_euler_axis,
    const std::vector<int> & parent_body_index)
{
    m_flagReRooted = false;
    size_t body_size = body_mass.size();
    dCASSERT(body_size == body_inertia.size());
    dCASSERT(body_size == body_position.size());
    dCASSERT(body_size == body_rotation.size());
    dCASSERT(body_size == parent_joint_dof.size());
    dCASSERT(body_size == parent_joint_pos.size());
    dCASSERT(body_size == parent_joint_euler_order.size());
    dCASSERT(body_size == parent_joint_euler_axis.size());
    dCASSERT(body_size == parent_body_index.size());

    this->initialize_resize(body_size);

    m_Ms = body_mass;
    m_Is = body_inertia;
    m_parent = parent_body_index;
    m_jointOrientMatrix = parent_joint_euler_axis;
    parent_joint_euler = parent_joint_euler_order;
    for (int i = 0; i < body_size; i++) m_order[i] = i;

    // prepare rotate axies
    int ndof = 3;   // root position
    for (size_t i = 0; i < body_size; ++i)
    {
        int jdof = parent_joint_dof[i];
        int jointType = joint_euler_str_to_joint_type(parent_joint_euler_order[i]);
        set_euler_callback(static_cast<int>(i), jointType);
        m_jointAxies[i].resize(jdof);
        for (int j = 0; j < jdof; ++j)
        {
            m_jointAxies[i][j] = parent_joint_euler_axis[i].row(size_t(parent_joint_euler_order[i][j] - 'X')); // TODO: row or column???
        }
        m_JwJoint[i].resize(jdof, Vector3d::Zero());
        m_JwdotJoint[i].resize(jdof, Vector3d::Zero());
        m_jointQIndex[i] = ndof;
        ndof += jdof;
        const auto& pjoint_pos = parent_joint_pos[i];
        m_cls[i] = body_rotation[i].transpose() * (- pjoint_pos + body_position[i]);
        if (parent_body_index[i] != -1)
        {
            m_dcls[i] = body_rotation[parent_body_index[i]].transpose() * (pjoint_pos - body_position[parent_body_index[i]]);
        }
    }

    m_q.resize(ndof, 0.0);
    m_qdot.resize(ndof, 0.0);
    m_qdotdot.resize(ndof, 0.0);

    // for re-root
    m_reOrder = m_order;
    m_reOrderBackMap = m_reOrder;
    m_reParent = m_parent;
    m_reCls = m_cls; // CoM in local
    m_reDcls = m_dcls; // from child joint to com
    m_reR = m_R;
}

CInverseDynamics::~CInverseDynamics(void)
{
    
}

void CInverseDynamics::initialize_resize(size_t body_count)
{
    this->m_jointAxies.resize(body_count);
    this->m_Is.resize(body_count, Matrix3d::Zero());
    this->m_Ms.resize(body_count, 0.0);
    this->m_cls.resize(body_count, Vector3d::Zero());
    this->m_dcls.resize(body_count, Vector3d::Zero());
    this->m_order.resize(body_count, -1);
    this->m_parent.resize(body_count, -1);

    this->m_jointEulerFactors.resize(body_count, NULL); // TODO: maybe we should use std::function or other wrapper...
    this->m_jointEulerMakers.resize(body_count, NULL);

    this->m_localLinVels.resize(body_count, Vector3d::Zero());
    this->m_localAngVels.resize(body_count, Vector3d::Zero());
    this->m_localLinAccels.resize(body_count, Vector3d::Zero());
    this->m_localAngAccels.resize(body_count, Vector3d::Zero());
    this->m_localForces.resize(body_count, Vector3d::Zero());
    this->m_localTorques.resize(body_count, Vector3d::Zero());
    this->m_jointQIndex.resize(body_count, 0);

    this->m_JwJoint.resize(body_count);
    this->m_JwdotJoint.resize(body_count);
    this->m_R.resize(body_count, Matrix3d::Identity());
    this->m_R0.resize(body_count, Matrix3d::Identity());
    this->m_jointOrientMatrix.resize(body_count, Matrix3d::Identity());
}

void CInverseDynamics::set_euler_callback(int i, int jointType)
{
    switch (jointType)
    {
    case 1:
    case 12:
    case 123:
        m_jointEulerFactors[i] = &EulerFactorXYZ;
        m_jointEulerMakers[i] = &EulerMakerXYZ;
        break;
    case 2:
    case 23:
    case 231:
        m_jointEulerFactors[i] = &EulerFactorYZX;
        m_jointEulerMakers[i] = &EulerMakerYZX;
        break;
    case 3:
    case 31:
    case 312:
        m_jointEulerFactors[i] = &EulerFactorZXY;
        m_jointEulerMakers[i] = &EulerMakerZXY;
        break;
    case 21:
    case 213:
        m_jointEulerFactors[i] = &EulerFactorYXZ;
        m_jointEulerMakers[i] = &EulerMakerYXZ;
        break;
    case 32:
    case 321:
        m_jointEulerFactors[i] = &EulerFactorZYX;
        m_jointEulerMakers[i] = &EulerMakerZYX;
        break;
    case 13:
    case 132:
        m_jointEulerFactors[i] = &EulerFactorXZY;
        m_jointEulerMakers[i] = &EulerMakerXZY;
        break;
    default:
        printf("Joint %d, input Euler Order not supported.\n", i);
        dCASSERT(false);
        std::exit(1);
    }
}


void CInverseDynamics::ConvertToGeneralizedCoordinates(
    const Vector3d& rootPos, const std::vector<Quaterniond>& vJointRots,
    std::vector<double>& q) const
{
    // root joint pos (3dof) + joint euler angle
    if (vJointRots.size() != m_parent.size())
    {
        std::cout << "In " << __func__ << ": vJointRots.size() != m_parent.size()" << std::endl;
        return;
    }

    q.resize(m_q.size(), 0);
    // root position
    for (int i = 0; i < 3; ++i)
        q[i] = rootPos[i];

    double a[3];
    size_t s = m_jointEulerFactors.size();
    Matrix3d mat;
    for (size_t i = 0; i < s; ++i)
    {
        mat = vJointRots[i].toRotationMatrix();
        a[0] = a[1] = a[2] = 0;
        m_jointEulerFactors[i]( // convert to parent local, 
            m_jointOrientMatrix[i].transpose() * mat* m_jointOrientMatrix[i], // if euler axis is arbitrarily orthogonal coordinate(not xyz), then should convert from xyz to this coordinate, then rotate, then convert back to xyz coordinate
            a[0], a[1], a[2]);
#if 0
        std::cout << __func__ << ", i = " << i << " a[0,1,2] = " << a[0] << " " << a[1] << " " << a[2] << std::endl;
#endif
        int qstart = m_jointQIndex[i];
        int qend = (int)m_q.size();
        if (i < s - 1)
            qend = m_jointQIndex[i + 1];
#if 0
        std::cout << "(qstart, qend) = (" << qstart << ", " << qend << ")" << std::endl;
#endif
        if (qend - qstart == 3 || qend - qstart == 1)
        {
            for (int qid = qstart; qid < qend; ++qid)
                q[qid] = a[qid - qstart];
        }
        else
        {
            std::cerr << "In" << __func__ << ", Line " << __LINE__ << "qend - qstart = " << qend - qstart << ". only 1 or 3 is supported. Exit.." << std::endl;
            std::exit(1);
        }
    }
}

void CInverseDynamics::ConvertToJointRotations(
    const std::vector<double>& q,
    Vector3d& rootPos, std::vector<Quaterniond>& vJointRots
) const
{

    if (q.size() != m_q.size())
    {
        std::cout << "In " << __func__ << "q.size() != m_q.size()" << std::endl;
        return;
    }

    vJointRots.resize(m_parent.size(), Quaterniond::Identity());

    // root position
    for (int i = 0; i < 3; ++i)
        rootPos[i] = q[i];

    double a[3] = { 0, 0, 0 };
    size_t s = m_jointEulerFactors.size();
    Matrix3d mat;
    for (size_t i = 0; i < s; ++i)
    {
        int qstart = m_jointQIndex[i];
        int qend = (int)m_q.size();
        if (i < s - 1)
            qend = m_jointQIndex[i + 1];

        for (int qid = qstart; qid < qend; ++qid)
            a[qid - qstart] = q[qid];
        for (int ai = qend - qstart; ai < 3; ++ai)
            a[ai] = 0;

        mat = m_jointEulerMakers[i](a[0], a[1], a[2]);
        mat = m_jointOrientMatrix[i] * mat * m_jointOrientMatrix[i].transpose(); // mat.TimesTranspose(m_jointOrientMatrix[i]);  
        // convert to local xyz, then rotate, then convert back to euler axis in parent space
        vJointRots[i] = Quaterniond(mat); // vJointRots[i].FromRotationMatrix(mat);
    }
}

void CInverseDynamics::ConvertToGeneralizedCoordinates(
    int bodyId, const Quaterniond& jointRot,
    double* q) const
{
    if (bodyId < 0 || bodyId >= (int)m_parent.size())
        return;

    int qstart = m_jointQIndex[bodyId];
    int qend = (int)m_q.size();
    if (bodyId < (int)m_parent.size() - 1)
        qend = m_jointQIndex[size_t(bodyId) + 1];

    double a[3];
    Matrix3d mat = jointRot.toRotationMatrix();
    m_jointEulerFactors[bodyId](m_jointOrientMatrix[bodyId].transpose() * mat * m_jointOrientMatrix[bodyId],
        a[0], a[1], a[2]);
    for (int qid = qstart; qid < qend; ++qid)
        q[qid - qstart] = a[qid - qstart];
}

Quaterniond CInverseDynamics::ConvertToJointRotation(
    const double* q, int bodyId
) const
{
    if (bodyId < 0 || bodyId >= (int)m_parent.size())
        return Quaterniond::Identity();

    int qstart = m_jointQIndex[bodyId];
    int qend = (int)m_q.size();
    if (bodyId < (int)m_parent.size() - 1)
        qend = m_jointQIndex[size_t(bodyId) + 1];

    double a[3] = { 0, 0, 0 };
    for (int qid = qstart; qid < qend; ++qid)
        a[qid - qstart] = q[qid - qstart];
    for (int ai = qend - qstart; ai < 3; ++ai)
        a[ai] = 0;

    Matrix3d mat = m_jointEulerMakers[bodyId](a[0], a[1], a[2]);
    mat = m_jointOrientMatrix[bodyId] * mat * m_jointOrientMatrix[bodyId].transpose();
    return Quaterniond(mat);
}

void CInverseDynamics::ConvertToGeneralizedCoordinates(
    int bodyId, const Quaterniond& jointRot,
    std::vector<double>& q) const
{
    if (bodyId < 0 || bodyId >= (int)m_parent.size())
    {
        std::cout << "In " << __func__ << "bodyId == " << bodyId << std::endl;
        return;
    }

    int qstart = m_jointQIndex[bodyId];
    int qend = (int)m_q.size();
    if (bodyId < (int)m_parent.size() - 1)
        qend = m_jointQIndex[size_t(bodyId) + 1];

    q.resize(size_t(qend) - size_t(qstart), 0);

    double a[3];
    Matrix3d mat = jointRot.toRotationMatrix();
    m_jointEulerFactors[bodyId](m_jointOrientMatrix[bodyId].transpose() * mat * m_jointOrientMatrix[bodyId],
        a[0], a[1], a[2]);
    for (int qid = qstart; qid < qend; ++qid)
        q[size_t(qid) - size_t(qstart)] = a[qid - qstart];
}

Quaterniond CInverseDynamics::ConvertToJointRotation(
    const std::vector<double>& q, int bodyId
) const
{
    if (bodyId < 0 || bodyId >= (int)m_parent.size())
    {
        std::cout << "In " << __func__ << "bodyId == " << bodyId << std::endl;
        return Quaterniond::Identity();
    }

    int qstart = m_jointQIndex[bodyId];
    int qend = (int)m_q.size();
    if (bodyId < (int)m_parent.size() - 1)
        qend = m_jointQIndex[size_t(bodyId) + 1];

    if (q.size() != size_t(qend) - size_t(qstart))
        return Quaterniond::Identity();

    double a[3] = {0, 0, 0};
    for (int qid = qstart; qid < qend; ++qid)
        a[qid - qstart] = q[size_t(qid) - size_t(qstart)];
    for (int ai = qend - qstart; ai < 3; ++ai)
        a[ai] = 0;

    Matrix3d mat = m_jointEulerMakers[bodyId](a[0], a[1], a[2]);
    mat = m_jointOrientMatrix[bodyId] * mat * m_jointOrientMatrix[bodyId].transpose();
    return Quaterniond(mat);
}

void CInverseDynamics::InitializeInverseDynamics(const std::vector<double>& q, 
    const std::vector<double>& qdot, const std::vector<double>& qdotdot)
{

    if (q.size() != m_q.size())
    {
        std::fill(m_q.begin(), m_q.end(), 0.0);
        std::cout << "In " << __func__ << "q.size() != m_q.size()" << std::endl;
    }
    else
    {
        m_q = q;
    }

    if (qdot.size() != m_qdot.size())
    {
        std::fill(m_qdot.begin(), m_qdot.end(), 0.0);
        std::cout << "In " << __func__ << "qdot.size() != m_qdot.size()" << std::endl;
    }
    else
    {
        m_qdot = qdot;
    }
    
    if (qdotdot.size() != m_qdotdot.size())
    {
        std::fill(m_qdotdot.begin(), m_qdotdot.end(), 0.0);
        std::cout << "In " << __func__ << "qdotdot.size() != m_qdotdot.size()" << std::endl;
    }
    else
    {
        m_qdotdot = qdotdot;
    }
    

    ///////////////////////////////////////
    // update rotation and jacobians
    size_t s = m_parent.size();
    for (size_t i = 0; i < s; ++i)
    {
        auto& axies = m_jointAxies[i];
        int qidbase = m_jointQIndex[i];

        auto& R = m_R[i];
        auto& JwJoint = m_JwJoint[i];
        auto& JwdotJoint = m_JwdotJoint[i];

        R = Eigen::AngleAxisd(m_q[qidbase], axies[0]).matrix(); // R = Matrix3d(axies[0], m_q[qidbase]); 
        JwJoint[0] = axies[0];
        JwdotJoint[0] = Vector3d::Zero();
        size_t dof = axies.size();
        Vector3d wr = JwJoint[0] * qdot[qidbase];

        for (size_t d = 1; d < dof; ++d)
        {
            JwJoint[d] = R * axies[d]; // local Jacobian
            JwdotJoint[d] = wr.cross(JwJoint[d]); // \dot{J_{\omega}}
            wr += JwJoint[d] * qdot[qidbase + d];  // local omega. \dot{R} =  
            R = R * Eigen::AngleAxisd(m_q[qidbase + d], axies[d]).matrix();  // R = R * Matrix3d(axies[d], m_q[qidbase + d]);
        }
        
        int pi = m_parent[i];
        if (pi >= 0)
        {
            m_R0[i] = m_R0[pi] * R;
        }
        else
        {
            m_R0[i] = R;
        }
    }
}

void CInverseDynamics::ComputeVelocityAcceleration(const Vector3d& gravity) // Karen Liu Tutorial, Page 22, Pass1.
{
    size_t s = m_parent.size();
    for (size_t i = 0; i < s; ++i)
    {
        int base = m_jointQIndex[i];
        auto& JwJoint = m_JwJoint[i];
        auto& JwdotJoint = m_JwdotJoint[i];
        auto& R = m_R[i];
        Matrix3d Rt = R.transpose();
        Vector3d angVelInParent = Vector3d::Zero();
        Vector3d angAccelInParent = Vector3d::Zero();
        for (size_t d = 0; d < JwJoint.size(); ++d)
        {
            angVelInParent += JwJoint[d] * m_qdot[base + d];  // angular velocity in parent coordinate
            angAccelInParent += JwdotJoint[d] * m_qdot[base + d] + JwJoint[d] * m_qdotdot[base + d];
        }

        if (i == 0) // root
        {
            m_localAngVels[0] = Rt * angVelInParent;  // in Karen Liu's Tutorial Page 23, Formula (81), Line 2
            m_localAngAccels[0] = Rt * angAccelInParent;  // in Karen Liu's Tutorial Page 23, Formula (81), Line 4
            m_localLinVels[0] = m_localAngVels[0].cross(m_cls[0]);  // in Karen Liu's Tutorial Page 23, Formula (81), Line 1
            m_localLinAccels[0] = m_localAngVels[0].cross(m_localLinVels[0]) + m_localAngAccels[0].cross(m_cls[0])  // in Karen Liu's Tutorial Page 23, Formula (81), Line 3 and Line 1
                - Rt * gravity;// gravity: in Karen Liu's Tutorial Page 24, Formula (85)

            // root has translation dofs
            m_localLinVels[0] += Rt * Vector3d(m_qdot[0], m_qdot[1], m_qdot[2]);
            m_localLinAccels[0] += Rt * Vector3d(m_qdotdot[0], m_qdotdot[1], m_qdotdot[2]);
        }
        else
        {
            int pi = m_parent[i];
            m_localAngVels[i] = Rt * (m_localAngVels[pi] + angVelInParent);  // in Karen Liu's Tutorial Page 22, Formula (78)
            m_localAngAccels[i] = Rt * (m_localAngAccels[pi] + angAccelInParent + m_localAngVels[pi].cross(angVelInParent));  // in Karen Liu's Tutorial Page 23, Formula (80)

            m_localLinVels[i] = Rt * (m_localLinVels[pi] + m_localAngVels[pi].cross(m_dcls[i])) +
                m_localAngVels[i].cross(m_cls[i]);  // in Karen Liu's Tutorial Page 23, Formula (79), Line 1

            m_localLinAccels[i] = Rt * (m_localLinAccels[pi] + m_localAngAccels[pi].cross(m_dcls[i])
                + m_localAngVels[pi].cross(m_localAngVels[pi].cross(m_dcls[i])))
                + m_localAngVels[i].cross(m_localAngVels[i].cross(m_cls[i]))
                + m_localAngAccels[i].cross(m_cls[i]);  // in Karen Liu's Tutorial Page 23, Formula (79), Line 6 and Line 7
        }
    }
}

void CInverseDynamics::ComputeForceTorque(
    const std::vector<Vector3d>& fext /*= std::vector<Vector3d>()*/,
    const std::vector<Vector3d>& text /*= std::vector<Vector3d>()*/
)
{
    //if (!fext.empty() && fext.size() != m_localForces.size())
    //    return;

    //if (!text.empty() && text.size() != m_localTorques.size())
    //    return;

    // initialize all forces and torques to zero
    std::fill(m_localForces.begin(), m_localForces.end(), Vector3d::Zero());
    std::fill(m_localTorques.begin(), m_localTorques.end(), Vector3d::Zero());
    if (m_flagReRooted)
        ReComputeR();

    // compute torques reversely from end-effector
    auto& parent = m_flagReRooted ? m_reParent : m_parent;
    auto& order = m_flagReRooted ? m_reOrder : m_order;
    auto& cls = m_flagReRooted ? m_reCls : m_cls;
    auto& dcls = m_flagReRooted ? m_reDcls : m_dcls;
    auto& R = m_flagReRooted ? m_reR : m_R;

    for (size_t i = order.size(); i > 0;)
    {
        --i;
        int bid = order[i];
        m_localForces[bid] += m_Ms[bid] * m_localLinAccels[bid];
        if (!fext.empty())
            m_localForces[bid] -= m_R0[bid].transpose() * fext[bid];

        m_localTorques[bid] += cls[i].cross(m_localForces[bid]) // given by local force
            + m_Is[bid] * m_localAngAccels[bid]  // Newton-Euler Equation. Karen Liu's Tutorial, Page 7, Formula (23), Line 5.
            + m_localAngVels[bid].cross(m_Is[bid] * m_localAngVels[bid]);  // or Karen Liu's Tutorial, Page 21, Formula (73)

        if (!text.empty())
            m_localTorques[bid] -= m_R0[bid].transpose() * text[bid];

        int pi = parent[i];
        if (pi >= 0)
        {
            int pid = order[pi];
            Vector3d fInParent = R[i] * m_localForces[bid];
            m_localForces[pid] += fInParent;  // add force to parent. Karen Liu's Tutorial, Page 21, Formula (72)
            m_localTorques[pid] += R[i] * m_localTorques[bid] + dcls[i].cross(fInParent);  // Karen Liu's Tutorial, Page 21, Formula (73)
        }
    }
}

void CInverseDynamics::ReRoot(int bodyId)
{
    if (bodyId < 0 || bodyId >= (int)m_parent.size())
    {
        printf("Warning! Invalid bodyId %d\n", bodyId);
        return;
    }

    // revise the hierarchy
    std::vector<bool> visited(m_parent.size(), false);
    m_reParent[0] = -1;
    m_reOrder[0] = bodyId;
    m_reOrderBackMap[bodyId] = 0;
    m_reCls[0] = Vector3d::Zero();
    m_reDcls[0] = Vector3d::Zero();
    visited[bodyId] = true;
    int id = 1;
    int bid = bodyId;
    int pid = m_parent[bid];
    // reverse the chain from the old root to the new root
    while (pid >= 0)
    {
        m_reParent[id] = id - 1;
        m_reOrder[id] = pid;
        m_reOrderBackMap[pid] = id;
        m_reCls[id] = -m_dcls[bid];
        m_reDcls[id] = -m_cls[bid];

        visited[pid] = true;

        bid = pid;
        pid = m_parent[bid];
        ++id;
    }

    // copy the unmodified part
    for (size_t i = 0; i < m_parent.size(); ++i)
    {
        if (!visited[i])
        {
            m_reOrder[id] = (int)i;
            m_reOrderBackMap[i] = id;
            m_reParent[id] = m_reOrderBackMap[m_parent[i]];
            m_reCls[id] = m_cls[i];
            m_reDcls[id] = m_dcls[i];

            id++;
        }
    }

    m_flagReRooted = true;
}

void CInverseDynamics::ReComputeR(void)
{
    m_reR[0] = Matrix3d::Identity();
    for (size_t i = 1; i < m_reOrder.size(); ++i)
    {
        if (m_reOrder[i] < m_reOrder[m_reParent[i]])  // a revsered joint
            m_reR[i] = m_R[m_reOrder[m_reParent[i]]].transpose();
        else
            m_reR[i] = m_R[m_reOrder[i]];
    }
}

// Parameter pos is output...
void CInverseDynamics::ComputeBodyPositions(std::vector<Vector3d>& pos) const // forward kinematics
{
    pos.resize(m_Ms.size(), Vector3d::Zero());
    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        if (m_parent[i] < 0)
            pos[i] = Vector3d(m_q[0], m_q[1], m_q[2]) + m_R0[i] * m_cls[i];
        else
            pos[i] = pos[m_parent[i]] + m_R0[m_parent[i]] * m_dcls[i] + m_R0[i] * m_cls[i];  // parent body's pos + joint's offset to parent body + child body's offset to joint
    }
}

void CInverseDynamics::ComputeCoMMomentums(
    double& mass, Vector3d& com,
    Vector3d& linVelocity, Vector3d& angMomentum,
    Matrix3d& inertia
) const
{
    mass = 0;
    linVelocity = Vector3d::Zero();
    angMomentum = Vector3d::Zero();
    com = Vector3d::Zero();

    std::vector<Vector3d> pos(m_Ms.size(), Vector3d::Zero());

    std::vector<Vector3d> lvels = m_localLinVels;
    std::vector<Vector3d> avels = m_localAngVels;
    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        mass += m_Ms[i];
        lvels[i] = m_R0[i] * lvels[i]; // global linear velocity
        avels[i] = m_R0[i] * avels[i];  // global angular velocity
        if (m_parent[i] < 0)
            pos[i] = Vector3d(m_q[0], m_q[1], m_q[2]) + m_R0[i] * m_cls[i];
        else
            pos[i] = pos[m_parent[i]] + m_R0[m_parent[i]] * m_dcls[i] + m_R0[i] * m_cls[i];  // parent body's pos + joint's offset to parent body + child body's offset to joint

        com += m_Ms[i] * pos[i];  // center of mass
        linVelocity += m_Ms[i] * lvels[i];  // global Linear Momentum
    }

    linVelocity /= mass;
    com /= mass;

    inertia = Matrix3d::Zero();

    // I_i = R_i I_i0 R_i'
    // H = sum(I_i w_i + cc_i X m v_i)
    // I = sum(trans(I_i, c_i));
    for (size_t i = 0; i < m_Ms.size(); i++)
    {
        Matrix3d I = m_R0[i] * m_Is[i] * m_R0[i].transpose(); // in global coordinate
        Vector3d cci = pos[i] - com;

        angMomentum += I * avels[i] + m_Ms[i] * cci.cross(lvels[i]);  // Rotation around body's com + Rotation around system's com

        TransInertia(I, m_Ms[i], cci.x(), cci.y(), cci.z());
        inertia += I; // AddInertia(inertia, I);  // total interia
    }
}

void CInverseDynamics::ComputeLocalW_q(std::vector<Vector3d>& Jwq, int bodyId) const
{
    if (bodyId < 0 || bodyId >= (int)m_parent.size())
    {
        printf("Warning! Invalid bodyId %d\n", bodyId);
        return;
    }

    auto axies = m_jointAxies[bodyId];
    int qidbase = m_jointQIndex[bodyId];

    Matrix3d R = Matrix3d::Identity();
    for (size_t d = 1; d < axies.size(); ++d)
    {
        R = Eigen::AngleAxisd(m_q[qidbase + d - 1], m_jointAxies[bodyId][d - 1]).matrix() * R;
        axies[d] = R * axies[d];
    }

    Jwq.resize(axies.size(), Vector3d::Zero());
    for (size_t d = 0; d < axies.size(); ++d)
    {
        for (size_t dj = d + 1; dj < axies.size(); ++dj)
            Jwq[d] += axies[d].cross(axies[dj]) * m_qdot[qidbase + dj]; // what's this
    }
}

void CInverseDynamics::ComputeMomentumRootJacobians(
    std::vector<Vector3d>& Jvq, std::vector<Vector3d>& JLq,
    std::vector<Vector3d>& Jvqdot, std::vector<Vector3d>& JLqdot
) const
{
    double mass = 0;
    Vector3d com = Vector3d::Zero();

    std::vector<Vector3d> pos(m_Ms.size(), Vector3d::Zero());
    std::vector<Vector3d> jpos(m_Ms.size(), Vector3d::Zero());

    // compute body position and joint position
    for (size_t i = 0; i < m_Ms.size(); ++i)  // forward kinematics
    {
        mass += m_Ms[i];
        if (m_parent[i] < 0)
            jpos[i] = Vector3d(m_q[0], m_q[1], m_q[2]);
        else
            jpos[i] = pos[m_parent[i]] + m_R0[m_parent[i]] * m_dcls[i];

        pos[i] = jpos[i] + m_R0[i] * m_cls[i];

        com += m_Ms[i] * pos[i];
    }
    com /= mass;

    // compute R0 for each dof
    std::vector<Matrix3d> R0s(m_q.size(), Matrix3d::Identity());
    std::vector<Vector3d> qaxies(m_q.size(), Vector3d::Zero());  // euler axis in global coordinate
    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        auto& axies = m_jointAxies[i];
        int qidbase = m_jointQIndex[i];
        size_t dof = axies.size();
        if (m_parent[i] >= 0)
            R0s[qidbase] = m_R0[m_parent[i]];
        qaxies[qidbase] = R0s[qidbase] * axies[0];

        for (size_t d = 1; d < dof; ++d)
        {
            R0s[d + qidbase] = R0s[d + qidbase - 1] * Eigen::AngleAxisd(m_q[d + qidbase - 1], axies[d - 1]).matrix();
            qaxies[d + qidbase] = R0s[d + qidbase] * axies[d];
        }
    }

    // compute Jvi and Jwi
    std::vector<std::vector<Vector3d>> Jvqs(m_Ms.size(), std::vector<Vector3d>(m_q.size(), Vector3d::Zero()));
    std::vector<std::vector<Vector3d>> Jwqs(m_Ms.size(), std::vector<Vector3d>(m_q.size(), Vector3d::Zero()));
    // compute Jvi/dqi and Jwi/dqi
    std::vector<std::vector<std::vector<std::vector<double>>>>
        Jvq_qs(m_Ms.size(), std::vector<std::vector<std::vector<double>>>(m_q.size(),
            std::vector<std::vector<double>>(3, std::vector<double>(m_q.size(), 0.0))));
    std::vector<std::vector<std::vector<std::vector<double>>>>
        Jwq_qs(m_Ms.size(), std::vector<std::vector<std::vector<double>>>(m_q.size(),
            std::vector<std::vector<double>>(3, std::vector<double>(m_q.size(), 0.0))));

    std::vector<Matrix3d> Is(m_Ms.size(), Matrix3d::Zero());
    std::vector<std::vector<Matrix3d>> I_qs(m_Ms.size(), std::vector<Matrix3d>(m_q.size(), Matrix3d::Zero()));
    //std::vector<std::vector<Matrix3d>> Pos_qs(m_Ms.size(), std::vector<Matrix3d>(m_q.size(), Matrix3d::ZERO));

    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        auto& Jvqi = Jvqs[i];
        auto& Jwqi = Jwqs[i];

        auto& Jvq_qi = Jvq_qs[i];
        auto& Jwq_qi = Jwq_qs[i];


        Is[i] = m_R0[i] * m_Is[i] * m_R0[i].transpose();  // interia in global coordinate
        auto& I_qi = I_qs[i];

        //auto &Pos_qi = Pos_qs[i];

        // root
        for (int j = 0; j < 3; ++j)
            Jvqi[j][j] = 1.0;  // root position (or linear velocity?)

        for (size_t j = 0; j < m_Ms.size(); ++j)
        {
            // make sure that body j is an ancestor of body i
            {  // This for loop may be simplified by parent table..
                int p = (int)i;
                while (p >= 0 && p != j)
                    p = m_parent[p];
                if (p < 0)
                    continue;
            }

            auto& axies = m_jointAxies[j];
            int qidbase = m_jointQIndex[j];
            size_t dof = axies.size();

            Vector3d fromJoint = pos[i] - jpos[j]; // in global coordinate

            for (size_t d = 0; d < dof; ++d)
            {
                Jwqi[qidbase + d] = qaxies[qidbase + d];//R0s[qidbase + d] * axies[d]/*m_JwJoint[j][d]*/;  // Karen Liu's Tutorial, Page 14, Formula (52)
                Jvqi[qidbase + d] = Jwqi[qidbase + d].cross(fromJoint);  // in Robot Dynamics Lecture Notes - ETH Z (RD2016script.pdf), Page 38, Formula (2.163)

                Matrix3d dRdq;
                dRdq << 0.0, -qaxies[qidbase + d][2], qaxies[qidbase + d][1],
                       qaxies[qidbase + d][2], 0.0, -qaxies[qidbase + d][0],
                       -qaxies[qidbase + d][1], qaxies[qidbase + d][0], 0.0; 
                // Karen Liu's Tutorial, Page 6, Formula (20). convert vector to matrix for cross

                I_qi[qidbase + d] = dRdq * Is[i] + Is[i] * dRdq.transpose();  // what's this..may be in Karen Liu Page 7, Formula (23), Line 4?
            }

            for (size_t d = 0; d < dof; ++d)
            {
                auto& Jvq_qiq = Jvq_qi[qidbase + d];
                auto& Jwq_qiq = Jwq_qi[qidbase + d];

                for (size_t k = 0; k < m_Ms.size(); ++k)
                {
                    // make sure that body k is an ancestor of body i
                    {
                        int p = (int)i;
                        while (p >= 0 && p != k)
                            p = m_parent[p];
                        if (p < 0)
                            continue;
                    }

                    bool isJAncestor = true;
                    {
                        int p = m_parent[j];
                        while (p >= 0 && p != k)
                            p = m_parent[p];
                        if (p < 0)
                            isJAncestor = false;
                    }

                    auto& axies2 = m_jointAxies[k];
                    int qidbase2 = m_jointQIndex[k];
                    size_t dof2 = axies2.size();
                    Vector3d fromJoint2 = pos[i] - jpos[k];

                    for (size_t d2 = 0; d2 < dof2; ++d2)
                    {
                        if (isJAncestor || (j == k && d2 < d))
                        {
                            Vector3d axies_q = qaxies[qidbase2 + d2].cross(Jwqi[qidbase + d]);

                            Jwq_qiq[0][d2 + qidbase2] = axies_q[0];
                            Jwq_qiq[1][d2 + qidbase2] = axies_q[1];
                            Jwq_qiq[2][d2 + qidbase2] = axies_q[2];

                            Vector3d temp = axies_q.cross(fromJoint) + qaxies[qidbase + d].cross(qaxies[qidbase2 + d2].cross(fromJoint));

                            Jvq_qiq[0][d2 + qidbase2] = temp[0];
                            Jvq_qiq[1][d2 + qidbase2] = temp[1];
                            Jvq_qiq[2][d2 + qidbase2] = temp[2];

                        }
                        else
                        {
                            // k is i's ancestor, but is not j's ancestor
                            // has no effect on Jwq
                            //Jwq_qiq[0][d2 + qidbase2];
                            //Jwq_qiq[1][d2 + qidbase2];
                            //Jwq_qiq[2][d2 + qidbase2];

                            Vector3d temp = qaxies[qidbase + d].cross(qaxies[qidbase2 + d2].cross(fromJoint2));
                            Jvq_qiq[0][d2 + qidbase2] = temp[0];
                            Jvq_qiq[1][d2 + qidbase2] = temp[1];
                            Jvq_qiq[2][d2 + qidbase2] = temp[2];
                        }
                    }
                }
            }

        }
    }

    // (\partial v_c / \partial qdot_root)
    Jvqdot.resize(m_q.size(), Vector3d::Zero());

    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        for (size_t d = 0; d < m_q.size(); ++d)
            Jvqdot[d] += Jvqs[i][d] * m_Ms[i];
    }
    for (size_t i = 0; i < m_q.size(); ++i)
        Jvqdot[i] /= mass;  // def of CoM: 

    // (\partial v_c / \partial q_root)
    Jvq.resize(m_q.size(), Vector3d::Zero());
    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        auto& Jvq_qi = Jvq_qs[i];
        auto& m = m_Ms[i];
        for (size_t d = 0; d < m_q.size(); ++d)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (size_t dj = 0; dj < m_q.size(); ++dj)
                    Jvq[d][j] += (Jvq_qi[dj][j][d] * m_qdot[dj]) * m;
            }
        }
    }
    for (size_t i = 0; i < m_q.size(); ++i)
        Jvq[i] /= mass;

    // (\partial L / \partial qdot_root)
    JLqdot.resize(m_q.size(), Vector3d::Zero());
    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        //Matrix3d Ii = m_R0[i] * m_Is[i].TimesTranspose(m_R0[i]);
        Matrix3d Ii = Is[i];
        Vector3d cci = pos[i] - com;
        Matrix3d ccix;
        ccix << 0.0, -cci[2], cci[1],
                cci[2], 0.0, -cci[0],
                -cci[1], cci[0], 0.0;

        for (size_t d = 0; d < m_q.size(); ++d)
            JLqdot[d] += Ii * Jwqs[i][d] + m_Ms[i] * (ccix * Jvqs[i][d]);
    }

    // (\partial L / \partial q_root)
    JLq.resize(m_q.size(), Vector3d::Zero());
    for (size_t i = 0; i < m_Ms.size(); ++i)
    {
        //Matrix3d Ii = m_R0[i] * m_Is[i].TimesTranspose(m_R0[i]);
        Matrix3d& Ii = Is[i];
        Vector3d cci = pos[i] - com;
        Matrix3d ccix;
        ccix << 0.0, -cci[2], cci[1],
             cci[2], 0.0, -cci[0],
            -cci[1], cci[0], 0.0;

        auto& Jvq_qi = Jvq_qs[i];
        auto& Jwq_qi = Jwq_qs[i];
        auto& m = m_Ms[i];
        auto& Ii_qs = I_qs[i];
        auto& Jwqi = Jwqs[i];
        auto& Jvqi = Jvqs[i];

        for (size_t d = 0; d < m_q.size(); ++d)
        {
            // \partial Ii/\partial qd * Jwij * qj
            auto& Ii_qd = Ii_qs[d];
            Vector3d temp = Vector3d::Zero();
            for (size_t dj = 0; dj < m_q.size(); ++dj)
                temp += Jwqi[dj] * m_qdot[dj];
            temp = Ii_qd * temp;

            JLq[d] += temp;

            // Ii * Jwij_qd * qj
            temp = Vector3d::Zero();
            for (size_t dj = 0; dj < m_q.size(); ++dj)
            {
                Vector3d temp2(Jwq_qi[dj][0][d], Jwq_qi[dj][1][d], Jwq_qi[dj][2][d]);
                temp2 *= m_qdot[dj];
                temp += temp2;
            }
            temp = Ii * temp;

            JLq[d] += temp;

            // \partial c / \partial qd x Jv_ij qj
            temp = Vector3d::Zero();
            for (size_t dj = 0; dj < m_q.size(); ++dj)
                temp += Jvqi[dj] * m_qdot[dj];
            // Jcq = Jvqdot*
            Vector3d temp2 = Jvqi[d] - Jvqdot[d];
            Matrix3d mat_temp2;
            mat_temp2 << 0.0, -temp2[2], temp2[1],
                temp2[2], 0.0, -temp2[0],
                -temp2[1], temp2[0], 0.0;
            temp = mat_temp2 * temp;

            JLq[d] += temp * m;

            // ci x Jvij_qd * qj
            temp = Vector3d::Zero();
            for (size_t dj = 0; dj < m_q.size(); ++dj)
            {
                Vector3d temp2(Jvq_qi[dj][0][d], Jvq_qi[dj][1][d], Jvq_qi[dj][2][d]);
                temp2 *= m_qdot[dj];
                temp += temp2;
            }
            temp = ccix * temp;
            JLq[d] += temp * m;
        }
    }
}
