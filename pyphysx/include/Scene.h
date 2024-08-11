/**
 * Copyright (c) CTU  - All Rights Reserved
 * Created on: 4/30/20
 *     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
 */

#ifndef SIM_PHYSX_SCENE_H
#define SIM_PHYSX_SCENE_H

#include <Physics.h>
#include <BasePhysxPointer.h>
#include <RigidDynamic.h>
#include "RigidStatic.h"
#include "Aggregate.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct BodyInfoState
{
    py::array_t<float> pos, quat, linear_vel, angular_vel;
};

class Scene : public BasePhysxPointer<physx::PxScene> {
public:
    Scene(const physx::PxFrictionType::Enum &friction_type,
          const physx::PxBroadPhaseType::Enum &broad_phase_type,
          const std::vector<physx::PxSceneFlag::Enum> &scene_flags,
          size_t gpu_max_num_partitions,
          float gpu_dynamic_allocation_scale
    ) : BasePhysxPointer() {
        physx::PxSceneDesc sceneDesc(Physics::get().physics->getTolerancesScale());
        sceneDesc.cpuDispatcher = Physics::get().dispatcher;
        sceneDesc.cudaContextManager = Physics::get().cuda_context_manager;
        sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;
        sceneDesc.gravity = physx::PxVec3(0.0f, 0.0f, -9.81f);
        for (const auto &flag : scene_flags) {
            sceneDesc.flags |= flag;
        }
        sceneDesc.frictionType = friction_type;
        sceneDesc.broadPhaseType = broad_phase_type;
        sceneDesc.gpuMaxNumPartitions = gpu_max_num_partitions;
        sceneDesc.gpuDynamicsConfig.patchStreamSize *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.forceStreamCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.contactBufferCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.contactStreamSize *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.foundLostPairsCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.constraintBufferCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.heapCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.tempBufferCapacity *= gpu_dynamic_allocation_scale;

        set_physx_ptr(Physics::get().physics->createScene(sceneDesc));
    }

    /** @brief Simulate scene for given amount of time dt and fetch results with blocking. */
    void simulate(float dt) {
        get_physx_ptr()->simulate(dt);
        get_physx_ptr()->fetchResults(true);
        simulation_time += dt;
    }

    void add_actor(RigidActor actor) {
        get_physx_ptr()->addActor(*actor.get_physx_ptr());
    }

    // Add by Zhenhua Song
    int get_num_static_actors() const
    {
        return get_physx_ptr()->getNbActors(physx::PxActorTypeFlag::eRIGID_STATIC);
    }

    auto get_static_rigid_actors()
    {
        int n = get_num_static_actors();
        std::vector<physx::PxRigidActor *> actors(n);
        get_physx_ptr()->getActors(physx::PxActorTypeFlag::eRIGID_STATIC,
                                   reinterpret_cast<physx::PxActor **>(&actors[0]), n);
        return from_vector_of_physx_ptr<RigidActor>(actors);
    }

    // Add by Zhenhua Song
    int get_num_dynamic_actors() const
    {
        return get_physx_ptr()->getNbActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC);
    }

    auto get_dynamic_rigid_actors()
    {
        int n = get_num_dynamic_actors();
        std::vector<physx::PxRigidDynamic *> actors(n);
        get_physx_ptr()->getActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC,
                                   reinterpret_cast<physx::PxActor **>(&actors[0]), n);
        return from_vector_of_physx_ptr<RigidDynamic, physx::PxRigidDynamic>(actors);
    }

    // Add by Zhenhua Song
    int get_num_articulation() const
    {
        return get_physx_ptr()->getNbArticulations();
    }

    // Add by Zhenhua Song
    auto get_articulations()
    {
        int n = get_num_articulation();
        std::vector<physx::PxArticulationBase*> result(n);
        get_physx_ptr()->getArticulations(reinterpret_cast<physx::PxArticulationBase**>(&result[0]), n);
        // return from_vector_of_physx_ptr(result);
    }

    // Add by Zhenhua Song
    auto get_num_constraints() const
    {
        return get_physx_ptr()->getNbConstraints();
    }

    void add_aggregate(Aggregate agg) {
        get_physx_ptr()->addAggregate(*agg.get_physx_ptr());
    }

    auto get_aggregates() {
        const auto n = get_physx_ptr()->getNbAggregates();
        std::vector<physx::PxAggregate *> aggs(n);
        get_physx_ptr()->getAggregates(&aggs[0], n);
        return from_vector_of_physx_ptr<Aggregate>(aggs);
    }

    // Add by Zhenhua Song
    auto get_gravity() const
    {
        return get_physx_ptr()->getGravity();
    }

    // Add by Zhenhua Song
    void set_gravity(float gx, float gy, float gz)
    {
        get_physx_ptr()->setGravity(physx::PxVec3(gx, gy, gz));
    }

    // Add by Zhenhua Song
    // get global position of rigid bodies
    auto getBodyPos()
    {
        if (this->dynamic_actors.size() == 0)
        {
            this->afterCreate();
        }
        const std::vector<RigidDynamic> & actors = this->dynamic_actors;
        py::array_t<float> pos(py::array::ShapeContainer({static_cast<int>(actors.size()), 3}));
        auto r = pos.mutable_unchecked<2>();
        for (int i = 0; i < actors.size(); i++)
        {
            auto pose = actors[i].get_global_pose();
            r(i, 0) = pose.p.x;
            r(i, 1) = pose.p.y;
            r(i, 2) = pose.p.z;
        }
        return pos;
    }

    // Add by Zhenhua Song
    // get global rotation of rigid bodies
    auto getBodyQuatScipy()
    {
        if (this->dynamic_actors.size() == 0)
        {
            this->afterCreate();
        }
        const std::vector<RigidDynamic>& actors = this->dynamic_actors;
        py::array_t<float> quat(py::array::ShapeContainer({ static_cast<int>(actors.size()), 4 }));
        auto r = quat.mutable_unchecked<2>();
        for (int i = 0; i < actors.size(); i++)
        {
            auto pose = actors[i].get_global_pose();
            r(i, 0) = pose.q.x;
            r(i, 1) = pose.q.y;
            r(i, 2) = pose.q.z;
            r(i, 3) = pose.q.w;
        }
        return quat;
    }

    // Add by Zhenhua Song
    void getBodyRot()
    {

    }

    // Add by Zhenhua Song
    // get global linear velocity of rigid bodies
    auto getBodyLinVel()
    {
        if (this->dynamic_actors.size() == 0)
        {
            this->afterCreate();
        }
        std::vector<RigidDynamic>& actors = this->dynamic_actors;
        py::array_t<float> vel(py::array::ShapeContainer({ static_cast<int>(actors.size()), 3 }));
        auto r = vel.mutable_unchecked<2>();
        for (int i = 0; i < actors.size(); i++)
        {
            auto v = actors[i].get_linear_velocity();
            r(i, 0) = v.x;
            r(i, 1) = v.y;
            r(i, 2) = v.z;
        }
        return vel;
    }

    // Add by Zhenhua Song
    // get global angular velocity of rigid bodies
    auto getBodyAngVel()
    {
        if (this->dynamic_actors.size() == 0)
        {
            this->afterCreate();
        }
        std::vector<RigidDynamic>& actors = this->dynamic_actors;
        py::array_t<float> vel(py::array::ShapeContainer({ static_cast<int>(actors.size()), 3 }));
        auto r = vel.mutable_unchecked<2>();
        for (int i = 0; i < actors.size(); i++)
        {
            auto v = actors[i].get_angular_velocity();
            r(i, 0) = v.x;
            r(i, 1) = v.y;
            r(i, 2) = v.z;
        }
        return vel;
    }

    // Add by Zhenhua Song
    auto GetBodyInfoState()
    {
        if (this->dynamic_actors.size() == 0)
        {
            this->afterCreate();
        }
        const std::vector<RigidDynamic>& actors = this->dynamic_actors;
        BodyInfoState state;
        state.pos = py::array_t<float>(py::array::ShapeContainer({ static_cast<int>(actors.size()), 3 }));
        state.quat = py::array_t<float>(py::array::ShapeContainer({ static_cast<int>(actors.size()), 4 }));
        auto r = state.pos.mutable_unchecked<2>(), q = state.quat.mutable_unchecked<2>();
        for (int i = 0; i < actors.size(); i++)
        {
            auto pose = actors[i].get_global_pose();
            r(i, 0) = pose.p.x;
            r(i, 1) = pose.p.y;
            r(i, 2) = pose.p.z;
            q(i, 0) = pose.q.x;
            q(i, 1) = pose.q.y;
            q(i, 2) = pose.q.z;
            q(i, 3) = pose.q.z;
        }
        state.linear_vel = this->getBodyLinVel();
        state.angular_vel = this->getBodyAngVel();
        return state;
    }

    // Add by Zhenhua Song
    auto getBodyMass()
    {
        if (this->dynamic_actors.size() == 0)
        {
            this->afterCreate();
        }
        std::vector<RigidDynamic>& actors = this->dynamic_actors;
        py::array_t<float> mass(static_cast<int>(actors.size()));
        auto r = mass.mutable_unchecked<1>();
        for (int i = 0; i < actors.size(); i++)
        {
            r(i) = actors[i].get_mass();
        }
    }

    // Add by Zhenhua Song
    auto getBodyInertia()
    {
        if (this->dynamic_actors.size() == 0)
        {
            this->afterCreate();
        }
        std::vector<RigidDynamic>& actors = this->dynamic_actors;
        
    }

    // Add by Zhenhua Song
    void afterCreate()
    {
        this->dynamic_actors = this->get_dynamic_rigid_actors();
        //this->dynamic_actors = this->get_dynamic_rigid_actors();
    }

public:
    double simulation_time = 0.;

// Add by Zhenhua Song
protected:
    std::vector<RigidDynamic> dynamic_actors;
    //std::vector<RigidActor> dynamic_actors;
};

#endif //SIM_PHYSX_SCENE_H
