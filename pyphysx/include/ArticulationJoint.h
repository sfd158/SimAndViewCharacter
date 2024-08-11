// Add by Zhenhua Song
#pragma once

#include <Physics.h>

class ArticulationJointRC : public BasePhysxPointer<physx::PxArticulationReducedCoordinate> {
    ArticulationJointRC(physx::PxArticulationLink * parent, const physx::PxTransform & linkPose)
    {
        auto j = Physics::get().physics->createArticulationReducedCoordinate();
        this->set_physx_ptr(j);
        physx::PxArticulationLink * link = j->createLink(parent, linkPose);
    }
};


class Articulation : public BasePhysxPointer<physx::PxArticulation> {
    Articulation()
    {
        auto j = Physics::get().physics->createArticulation();
        this->set_physx_ptr(j);

        
    }
};
