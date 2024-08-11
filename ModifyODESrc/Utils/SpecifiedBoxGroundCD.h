#pragma once
#include <ode/ode.h>

int ModifiedCollisionDetection(dGeomID o1, dGeomID o2, dContact *contact, int iMaxContact);
void SortContactByDepth(dContact * contact, int n);
int CollidGroundBoxBox(dGeomID g, dGeomID o, int n, dContact *cont);
int CollidGroundBoxSphere(dGeomID g, dGeomID o, int n, dContact *cont);
int CollidGroundBoxCapsule(dGeomID g, dGeomID o, int n, dContact *cont);
