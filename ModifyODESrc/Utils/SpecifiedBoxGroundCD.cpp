#include <SpecifiedBoxGroundCD.h>
#include <cmath>
#include <algorithm>

void SortContactByDepth(dContact * contact, int n)
{
    std::sort(contact, contact + n, [](dContact &lhs, dContact &rhs) { return abs(lhs.geom.depth) < abs(rhs.geom.depth); });
}

int CollidGroundBoxBox(dGeomID g, dGeomID o, int n, dContact *cont)
{
    if (n <= 3 || !cont)
        return 0;

    dIASSERT (dGeomGetClass(g) == dBoxClass);
    dIASSERT (dGeomGetClass(g) == dBoxClass);
    
    const dReal *gr = dGeomGetRotation(g);
    const dReal *gp = dGeomGetPosition(g);
    dVector3 gpos;
    dMULTIPLY1_331(gpos, gr, gp);
    dVector3 gside;
    dGeomBoxGetLengths(g, gside);
    gside[0] *= 0.5;
    gside[1] *= 0.5;
    gside[2] *= 0.5;
        
    const dReal *orient = dGeomGetRotation(o);
    const dReal *op = dGeomGetPosition(o);
    dVector3 oside;
    dGeomBoxGetLengths(o, oside);
    oside[0] *= 0.5;
    oside[1] *= 0.5;
    oside[2] *= 0.5;        

    dMatrix3 relr;
    dMULTIPLY1_333(relr, gr, orient);

    dVector3 opos;
    dMULTIPLY1_331(opos, gr, op);
    //dMULTIPLY0_331(oposg, relr, op);

    if (opos[1] < gpos[1] + gside[1])
        return -1;

    dVector3 offs[8] = {
        { oside[0],  oside[1],  oside[2] },
        { oside[0],  oside[1], -oside[2] },
        { oside[0], -oside[1],  oside[2] },
        { oside[0], -oside[1], -oside[2] },
        {-oside[0],  oside[1],  oside[2] },
        {-oside[0],  oside[1], -oside[2] },
        {-oside[0], -oside[1],  oside[2] },
        {-oside[0], -oside[1], -oside[2] }
    };
    dVector3 vertex;

    bool in[8] = {false};
    double depth[8] = {0.0};
    int incnt = 0;
    for (int i = 0; i < 8; i++)
    {
        in[i] = false;

        memcpy(vertex, opos, sizeof(dVector3));
        dMULTIPLYADD0_331(vertex, relr, offs[i]);

        if (vertex[1] - gpos[1] <= gside[1])
        {
            if (vertex[0] - gpos[0] <= gside[0] && vertex[0] - gpos[0] >= -gside[0] &&
                vertex[2] - gpos[2] <= gside[2] && vertex[2] - gpos[2] >= -gside[2])
            {
                in[i] = true;
                depth[i] = gside[1] - (vertex[1] - gpos[1]);
                ++incnt;
            }
            else
                return -1;
        }

        if (incnt >= 5 || incnt > n)
            return -1;
    }

    int idx = 0;
    for (int i = 0; i < 8; i++)
    {
        if (in[i])
        {
            cont[idx].geom.g1 = g;
            cont[idx].geom.g2 = o;
            cont[idx].geom.side1 = -1;
            cont[idx].geom.side2 = -1;
            cont[idx].geom.depth = depth[i];

            cont[idx].geom.normal[0] = -gr[1];
            cont[idx].geom.normal[1] = -gr[5];
            cont[idx].geom.normal[2] = -gr[9];

            memcpy(cont[idx].geom.pos, op, sizeof(dVector3));
            dMULTIPLYADD0_331(cont[idx].geom.pos, orient, offs[i]);
            idx++;
        }
    }

    dIASSERT(idx == incnt);
    return incnt;
}

int CollidGroundBoxSphere(dGeomID g, dGeomID o, int n, dContact *cont)
{
    if (n <= 0 || !cont)
        return 0;

    dIASSERT (dGeomGetClass(g) == dBoxClass);
    dIASSERT (dGeomGetClass(o) == dSphereClass);
    
    const dReal *gr = dGeomGetRotation(g);
    const dReal *gp = dGeomGetPosition(g);
    dVector3 gpos;
    dMULTIPLY1_331(gpos, gr, gp);
    dVector3 gside;
    dGeomBoxGetLengths(g, gside);
    gside[0] *= 0.5;
    gside[1] *= 0.5;
    gside[2] *= 0.5;

    const dReal *op = dGeomGetPosition(o);
    double r = dGeomSphereGetRadius(o);
    dVector3 opos;
    dMULTIPLY1_331(opos, gr, op);
    
    if (opos[1] < gpos[1] + gside[1])
        return -1;

    if (opos[1] - r > gpos[1] + gside[1])
        return 0;

    if (abs(opos[0] - gpos[0]) > gside[0] || abs(opos[2] - gpos[2]) > gside[2])
        return -1;


    cont[0].geom.g1 = g;
    cont[0].geom.g2 = o;
    cont[0].geom.side1 = -1;
    cont[0].geom.side2 = -1;
    cont[0].geom.depth = gpos[1] + gside[1] - (opos[1] - r);

    cont[0].geom.normal[0] = -gr[1];
    cont[0].geom.normal[1] = -gr[5];
    cont[0].geom.normal[2] = -gr[9];

    cont[0].geom.pos[0] = op[0] - gr[1] * r;
    cont[0].geom.pos[1] = op[1] - gr[5] * r;
    cont[0].geom.pos[2] = op[2] - gr[9] * r;

    return 1;
}

int CollidGroundBoxCapsule(dGeomID g, dGeomID o, int n, dContact *cont)
{
    if (n <= 1 || !cont)
        return 0;

    dIASSERT (dGeomGetClass(g) == dBoxClass);
    dIASSERT (dGeomGetClass(o) == dCapsuleClass);
    
    const dReal *gr = dGeomGetRotation(g);
    const dReal *gp = dGeomGetPosition(g);
    dVector3 gpos;
    dMULTIPLY1_331(gpos, gr, gp);
    dVector3 gside;
    dGeomBoxGetLengths(g, gside);
    gside[0] *= 0.5;
    gside[1] *= 0.5;
    gside[2] *= 0.5;

    const dReal *orient = dGeomGetRotation(o);
    const dReal *op = dGeomGetPosition(o);
    double r;
    double l;
    dGeomCapsuleGetParams(o, &r, &l);
    l *= 0.5;
    dVector3 opos;
    dMULTIPLY1_331(opos, gr, op);

    if (opos[1] < gpos[1] + gside[1])
        return -1;

    dVector3 alongg;
    dVector3 along = { orient[2], orient[6], orient[10] };
    dMULTIPLY1_331(alongg, gr, along);

    dVector3 oc1, oc2;
    oc1[0] = opos[0] + l * alongg[0];
    oc1[1] = opos[1] + l * alongg[1];
    oc1[2] = opos[2] + l * alongg[2];
    oc2[0] = opos[0] - l * alongg[0];
    oc2[1] = opos[1] - l * alongg[1];
    oc2[2] = opos[2] - l * alongg[2];

    bool cont1 = oc1[1] - r <= gpos[1] + gside[1];
    bool cont2 = oc2[1] - r <= gpos[1] + gside[1];
    
    if (!cont1 && !cont2)
        return 0;

    if ((cont1 && (abs(oc1[0] - gpos[0]) > gside[0] || abs(oc1[2] - gpos[2]) > gside[2])) ||
        (cont2 && (abs(oc2[0] - gpos[0]) > gside[0] || abs(oc2[2] - gpos[2]) > gside[2])))
        return -1;

    int cnt = 0;
    if (cont1)
    {
        cont[0].geom.g1 = g;
        cont[0].geom.g2 = o;
        cont[0].geom.side1 = -1;
        cont[0].geom.side2 = -1;
        cont[0].geom.depth = gpos[1] + gside[1] - (oc1[1] - r);

        cont[0].geom.normal[0] = -gr[1];
        cont[0].geom.normal[1] = -gr[5];
        cont[0].geom.normal[2] = -gr[9];

        cont[0].geom.pos[0] = op[0] + l * along[0] - gr[1] * r;
        cont[0].geom.pos[1] = op[1] + l * along[1] - gr[5] * r;
        cont[0].geom.pos[2] = op[2] + l * along[2] - gr[9] * r;

        ++cnt;
    }

    if (cont2)
    {
        int i = cont1 ? 1 : 0;
        cont[i].geom.g1 = g;
        cont[i].geom.g2 = o;
        cont[i].geom.side1 = -1;
        cont[i].geom.side2 = -1;
        cont[i].geom.depth = gpos[1] + gside[1] - (oc2[1] - r);

        cont[i].geom.normal[0] = -gr[1];
        cont[i].geom.normal[1] = -gr[5];
        cont[i].geom.normal[2] = -gr[9];

        cont[i].geom.pos[0] = op[0] - l * along[0] - gr[1] * r;
        cont[i].geom.pos[1] = op[1] - l * along[1] - gr[5] * r;
        cont[i].geom.pos[2] = op[2] - l * along[2] - gr[9] * r;

        ++cnt;
    }

    return cnt;
}

// Modified from Libin Liu's code..
int ModifiedCollisionDetection(dGeomID o1, dGeomID o2, dContact *contact, int iMaxContact)
{
    dBodyID b1 = dGeomGetBody(o1);
    dBodyID b2 = dGeomGetBody(o2);
    if (!b1 && !b2) // ground geometry
        return 0;

    // make sure that o1 is always the ground geometry
    if (b1 && !b2)
    {
        std::swap(o1, o2);
        std::swap(b1, b2);
    }

    int n = -1;
    {
        if (!b1)
        {
            switch (dGeomGetClass(o2))
            {
            case dBoxClass:
                n = CollidGroundBoxBox(o1, o2, iMaxContact, contact);
                break;
            case dSphereClass:
                n = CollidGroundBoxSphere(o1, o2, iMaxContact, contact);
                break;
            case dCapsuleClass:
                n = CollidGroundBoxCapsule(o1, o2, iMaxContact, contact);
                break;
            }
        }

        if (n > iMaxContact)
        {
            SortContactByDepth(contact, n);
            n = iMaxContact;
        }
    }

    if (n < 0)
        n = dCollide(o1, o2, iMaxContact, &(contact[0].geom), sizeof(dContact));

    return n;
}