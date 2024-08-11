#include  "JMJ.h"
#include<cstdlib>
#include<vector>
#include "Eigen/Sparse"
using namespace JMJ;
using namespace std;

void setOrder(const int num_body, const vector<treeNodeBase*>& myNode, vector<int>& order2Idx)
{
    std::vector<bool> flag(num_body,false);
    
    for(int i=num_body-1;i>=0;i--)
    {
        order2Idx.push_back(i);
        if (myNode[i]->parent != -1) order2Idx.push_back(myNode[i]->parent);
    }
}

void JMJBase::Reset(const int _body_num, const double* _mass, const int _ball_num, const int _hinge_num,
    const int* _bidx0, const int* _bidx1, const double* kds, const double* cfm, const double _dt)
{
    body_num = _body_num;
    ball_num = _ball_num;
    hinge_num = _hinge_num;
    mass.resize(_body_num);
    parent = std::vector<int>(body_num + ball_num + hinge_num, -1);
    jointchild.resize(ball_num + hinge_num);
    for (int i = 0; i < body_num; i++) mass[i] = _mass[i];
    for (int i=0; i<_ball_num+_hinge_num;i++)
    {
        parent[_bidx0[i]] = i + body_num;
        parent[i + body_num] = _bidx1[i];
        jointchild[i] = _bidx0[i];
    }

    dt = _dt;

    order2idx.clear();
    for(int i=body_num-1; i>=0;i--)
    {
        order2idx.push_back(i);
        if (parent[i] != -1)  order2idx.push_back(parent[i]);
    }

    joint_d.resize(6 * (ball_num + hinge_num));
    for(int i=0;i<ball_num;i++)
    {
        for (int j = 0; j < 3; j++)
        {
            joint_d[i * 6 + j] = -cfm[i * 3 + j]/dt;
            joint_d[i * 6 + j + 3] = -1.0 / (kds[i] * dt);
        }
    }
    for(int i=0;i<hinge_num;i++)
    {
        for (int j = 0; j < 5; j++) joint_d[ball_num * 6 + i * 6 + j] = -cfm[ball_num * 3 + i * 5 + j]/dt;
        joint_d[ball_num * 6 + i * 6 + 5] = -1.0/(kds[i + ball_num]*dt);
    }
}


void JMinvJ::_Reset(const int _num_body, const double* mass,const double* Inertia,
    const int num_joint,
    const double* Jlin0, const double* Jlin1,
    const double* Jang0, const double* Jang1,
    const int* bidx0, const int* bidx1,
    const double* R, const double* axis,
    const int* hinge_flag, const double* kds,
    const double* cfm, const double* c, const double dt
            )
{
    num_body = _num_body;
    num_node = num_body + num_joint;

    //std::cout << "build body" << std::endl;
    // build body node
    for(int i=0; i<num_body; i++){
        bodyNode* body = new bodyNode();
        myNode.push_back(body);
        body->setMass(mass[i], Inertia+9*i);
        body->idx = i;
    }
    int row_used = 0;
    int hinge_used = 0;
    for(int i=0; i<num_joint; i++){

        // build node first
        jointNode* joint = new jointNode();
        bodyNode* bChild = static_cast<bodyNode*>(myNode[bidx0[i]]);

        // set idx
        myNode.push_back(joint);
        joint->idx = static_cast<int>(myNode.size())-1;

        // check parent is NULL?
        if (bidx1[i] != -1) {
            bodyNode* bParent = static_cast<bodyNode*>(myNode[bidx1[i]]);
            joint->set_Parent(bParent);
        }

        // it must have a child anyway..
        bChild->set_Parent(joint);       

        // now set J and c
        if(hinge_flag[i]){
            joint->set_hinge(Jlin0+row_used*3,Jlin1+row_used*3,Jang0+row_used*3,Jang1+row_used*3,
                            axis+hinge_used*3,kds+row_used,cfm[i],dt, bChild);
            hinge_used+=1;
            row_used+=5;
        }else{
            joint->set_ball(Jlin0+row_used*3,Jlin1+row_used*3,Jang0+row_used*3,Jang1+row_used*3,
                            R+bidx1[i]*9,kds+row_used,cfm[i],dt, bChild);
            row_used+=3;
        }

    }

    setOrder(num_body, myNode, order2Idx);

#ifdef DEBUGJMJ
    
    A.resize(6*num_node,6*num_node);
    A.setZero();
    for(auto cur_node : myNode)
    {
        int idx = cur_node->idx;
        A.block(idx * 6, idx * 6, 6, 6) = cur_node->D;
        if(cur_node->parent!=-1)
        {
            int idx_p = myNode[cur_node->parent]->idx;
            A.block(idx * 6, idx_p * 6, 6, 6) = cur_node->J;
            A.block(idx_p * 6, idx * 6, 6, 6) = cur_node->J.transpose();
        }
    }
   
#endif
    // factor
    factorJMJ();
}

void JMJ::JMinvJ::_ResetWithBase(const double* Inertia, const double* JB0, const double* JB1, const double* JH0, const double* JH1, const double* R, const double* axis, const JMJBase* _base)
{

    num_body = _base->body_num;
    num_node = num_body + _base->hinge_num + _base->ball_num;
    for (int i = 0; i < num_body; i++) {
        bodyNode* body = new bodyNode();
        myNode.push_back(body);
        body->setMass(_base->mass[i], Inertia + 9 * i);
        body->idx = i;
        body->parent = _base->parent[i];
    }

    int ball_num = _base->ball_num;
    int hinge_num = _base->hinge_num;
    //std::cout << "lalal" << std::endl;
    for (int i = 0; i < ball_num + hinge_num; i++) {
        // build node first
        jointNode* joint = new jointNode();
        // set idx
        myNode.push_back(joint);
        joint->idx = i+num_body;
        joint->parent = _base->parent[i + num_body];
        //std::cout << "before set jacobian" << std::endl;
        
        if(i<ball_num)  joint->set_jacobian_fast( 3, JB0 + i * 3 * 6, JB1 + i * 3 * 6, R + joint->parent * 9, _base->joint_d.data() + i * 6, myNode[_base->jointchild[i]]);
        else   joint->set_jacobian_fast(5, JH0 + (i-ball_num) * 5 * 6, JH1 + (i-ball_num) * 5 * 6, axis + (i-ball_num) * 3, _base->joint_d.data() + i * 6, myNode[_base->jointchild[i]]);
    }

    //std::cout << "after set jacobian" << std::endl;
    order2Idx = _base->order2idx;
    //std::cout << "order2Idx ";
    //for (auto i : order2Idx) std::cout << i << " ";
    //std::cout << std::endl;
    //std::cout << "before factor" << std::endl;
#ifdef DEBUGJMJ

    A.resize(6 * num_node, 6 * num_node);
    A.setZero();
    for (auto cur_node : myNode)
    {
        int idx = cur_node->idx;
        A.block(idx * 6, idx * 6, 6, 6) = cur_node->D;
        if (cur_node->parent != -1)
        {
            int idx_p = myNode[cur_node->parent]->idx;
            A.block(idx * 6, idx_p * 6, 6, 6) = cur_node->J;
            A.block(idx_p * 6, idx * 6, 6, 6) = cur_node->J.transpose();
        }
    }

#endif
    factorJMJ();
}

void JMinvJ::factorJMJ(){
    //std::cout << "total" << myNode.size();
    auto Identity = Matrix66::Identity();
    for(auto cur_node_idx : order2Idx){
        auto *cur_node = myNode[cur_node_idx];
        //for(int cur_child = cur_node->child; cur_child!=-1; 
        //        cur_child = myNode[cur_child]->brother)
        //{
        //    cur_node->D.noalias() -= myNode[cur_child]->J.transpose() * myNode[cur_child]->D * myNode[cur_child]->J;    
        //}
       
        cur_node->Dinv.compute(cur_node->D);
        if(cur_node->parent!=-1){
            auto *cur_parent_node = myNode[cur_node->parent];
            Matrix66 J_tmp = cur_node->J;
            cur_node->J = cur_node->Dinv.solve(cur_node->J);
            cur_parent_node->D.noalias() -= cur_node->J.transpose() * J_tmp;//cur_node->D * cur_node->J;
        }
    }
}

void JMinvJ::solveX(double* _b, double* _lambda,const int col)
{

    const int row_x = num_body * 6;
    const int row_lambda = (num_node - num_body) * 6;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> x(_b,row_x,col);    
    
    // build lambda
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        lambda(_lambda, row_lambda, col);
    
#ifdef DEBUGJMJ
    Eigen::MatrixXd x_ = x;
    Eigen::MatrixXd lambda_ = lambda;
#endif

    //std::cout << "forward" << std::endl;
    // forward

    const int joint_bias = num_body*6;    
    for(auto idx: order2Idx){
        //std::cout << "idx: " << idx << std::endl;
        auto* cur_node = myNode[idx];

        if(cur_node->isConstraint){
            
            // joint's child is body
            //for(int cur_child_idx = cur_node->child; cur_child_idx!=-1; cur_child_idx=myNode[cur_child_idx]->brother){
            //    auto cur_child = myNode[cur_child_idx];
            //    lambda.block(cur_node->idx*6 - joint_bias,0,6,col).noalias() -= cur_child->J.transpose() * x.block(cur_child->idx*6,0,6,col);
            //}
            if(cur_node->parent !=-1)
            {
                auto *cur_parent_node = myNode[cur_node->parent];
                x.block(cur_parent_node->idx*6, 0, 6, col).noalias() -= cur_node->J.transpose() * lambda.block(cur_node->idx * 6 - joint_bias, 0, 6, col);
            }
        }

        else{
            // body row is in x
            //Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> cur_row(x.row(cur_node->idx*6).data(),6,col);

            // body's child is joint
            //for(int cur_child_idx = cur_node->child; cur_child_idx!=-1; cur_child_idx=myNode[cur_child_idx]->brother){
            //    auto cur_child = myNode[cur_child_idx];
            //    cur_row.noalias() -= cur_child->J.transpose() * lambda.block(cur_child->idx*6-joint_bias,0,6,col);
           // }
            if (cur_node->parent != -1)
            {
                auto* cur_parent_node = myNode[cur_node->parent];
                lambda.block(cur_parent_node->idx*6-joint_bias, 0, 6, col).noalias() -= cur_node->J.transpose() * x.block(cur_node->idx*6, 0, 6, col);
            }
        }

    }

    //std::cout << "backward" << std::endl;
    // backward

    for(auto idxP = order2Idx.rbegin(); idxP!=order2Idx.rend(); ++idxP ) {
        auto cur_node = myNode[*idxP];
        if(cur_node->isConstraint){
            
            // joint force is in lambda
            lambda.block(cur_node->idx*6 - joint_bias, 0, 6, col) = cur_node->Dinv.solve( lambda.block(cur_node->idx*6 - joint_bias, 0, 6, col) );
            if(cur_node->parent!=-1){
                auto cur_parent_node = myNode[cur_node->parent];
                // joint parent is a body
                lambda.block(cur_node->idx*6 - joint_bias, 0, 6, col).noalias() -= cur_node->J * x.block(cur_parent_node->idx*6,0,6,col);
            }
        }
        else
        {
            // body velocity is in x
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> cur_row(x.row(cur_node->idx*6).data(),6,col);
            cur_row = cur_node->Dinv.solve(cur_row);
            if(cur_node->parent!=-1){
                auto cur_parent_node = myNode[cur_node->parent];
                // body's parent is a joint
                cur_row.noalias() -= cur_node->J * lambda.block(cur_parent_node->idx*6- joint_bias,0,6,col);
            }           
        }
    }

#ifdef DEBUGJMJ
    Eigen::MatrixXd solu;
    solu.resize(myNode.size() * 6, col);
    solu.topRows(num_body * 6) = x;
    solu.bottomRows(row_lambda) = lambda;

    Eigen::MatrixXd rhs;
    rhs.resize(myNode.size() * 6, col);
    rhs.topRows(num_body * 6) = x_;
    rhs.bottomRows(row_lambda) = lambda_;
    
    std::cout <<"A" << std::endl << A << std::endl;
    std::cout << "Ax" << std::endl << A * solu << std::endl;
    std::cout << "b" << std::endl << rhs << std::endl;
    std::cout << "error: " << (A * solu - rhs).cwiseAbs().maxCoeff() << std::endl;
#endif

}

void JMinvJ::backward_solve(double* b, double* Lambda, double* G_b, double* G_Lambda, const int col, double* G_I, double* G_Jl0, double* G_Jl1, double* G_Ja0, double* G_Ja1, 
                    double* G_R, double* G_axis){
        

        // first, calculate l_b 
        solveX(G_b,G_Lambda,col);

        const int row_b = num_body *6;
        const int row_Lambda = (num_node - num_body) *6;
        const auto l_b = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> (G_b,row_b*6,col).sparseView();
        const auto l_Lambda = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> (G_Lambda,row_Lambda*6,col).sparseView();
        
        // then we can calculate l_A = - 0.5(l_b x^T + xl_b^T)
        
        // Eigen::Sparse l_A; 
        
}