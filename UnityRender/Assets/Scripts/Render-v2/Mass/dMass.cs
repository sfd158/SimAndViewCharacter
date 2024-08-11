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
using System;

namespace RenderV2
{
	/// <summary>
	/// Modified from Open Dynamics Engine
	/// </summary>
    public class DMass
    {
        public double mass = 0.0F;
		public double[, ] I = new double[3, 3];
		public double[] c = new double[3];
		// copied from Open Dynamics Engine..
		public void SetZero()
        {
			for(int i=0; i<3; i++)
            {
				for(int j=0; j<3; j++)
                {
					this.I[i, j] = 0;
                }
            }
			for(int i=0; i<3; i++)
            {
				this.c[i] = 0;
            }
			this.mass = 0;
        }

		public void SetSphere(double density, double radius)
		{
			SetSphereTotal(4.0 / 3.0 * Math.PI * radius * radius * radius * density, radius);
		}


		public void SetSphereTotal(double total_mass, double radius)
		{
			SetZero();
			this.mass = total_mass;
			double II = 0.4 * total_mass * radius * radius;
			this.I[0, 0] = II;
			this.I[1, 1] = II;
			this.I[2, 2] = II;
		}


		public void SetCapsule(double density, int direction, double radius, double length)
		{
			double M1, M2, Ia, Ib;
			if (!(direction >= 1 && direction <= 3))
            {
				throw new ArgumentException("bad direction number");
            }

			SetZero();
			M1 = Math.PI * radius * radius * length * density;              // cylinder mass
			M2 = 4.0 / 3.0 * Math.PI * radius * radius * radius * density; // total cap mass
			this.mass = M1 + M2;
			Ia = M1 * (0.25 * radius * radius + 1.0 / 12.0 * length * length) +
			  M2 * (0.4 * radius * radius + 0.375 * radius * length + 0.25 * length * length);
			Ib = (M1 * 0.5 + M2 * 0.4) * radius * radius;
			this.I[0, 0] = Ia;
			this.I[1, 1] = Ia;
			this.I[2, 2] = Ia;
			this.I[direction - 1, direction - 1] = Ib;
		}


		public void SetCapsuleTotal(double total_mass, int direction, double a, double b)
		{
			SetCapsule(1.0, direction, a, b);
			dMassAdjust(total_mass);
		}


		public void SetCylinder(double density, int direction, double radius, double length)
		{
			SetCylinderTotal(Math.PI * radius * radius * length * density, direction, radius, length);
		}

		public void SetCylinderTotal(double total_mass, int direction, double radius, double length)
		{
			double r2, I;
			if (!(direction >= 1 && direction <= 3))
			{
				throw new ArgumentException("bad direction number");
			}
			SetZero();
			r2 = radius * radius;
			mass = total_mass;
			I = total_mass * (0.25 * r2 + 1.0 / 12.0 * length * length);
			this.I[0, 0] = I;
			this.I[1, 1] = I;
			this.I[2, 2] = I;
			this.I[direction - 1, direction - 1] = total_mass * 0.5 * r2;
		}


		public void SetBox(double density, double lx, double ly, double lz)
		{
			SetBoxTotal(lx * ly * lz * density, lx, ly, lz);
		}


		public void SetBoxTotal(double total_mass, double lx, double ly, double lz)
		{
			SetZero();
			this.mass = total_mass;
			this.I[0, 0] = total_mass / 12.0 * (ly * ly + lz * lz);
			this.I[1, 1] = total_mass / 12.0 * (lx * lx + lz * lz);
			this.I[2, 2] = total_mass / 12.0 * (lx * lx + ly * ly);
		}

		public void dMassAdjust(double newmass)
		{
			double scale = newmass / this.mass;
			this.mass = newmass;
			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) this.I[i, j] *= scale;
		}

		public void Add(ref DMass b)
        {
			double denom = 1.0 / (this.mass + b.mass);
			for (int i = 0; i < 3; i++)
			{
				this.c[i] = (this.c[i] * this.mass + b.c[i] * b.mass) * denom;
			}
			this.mass += b.mass;
			for(int i=0; i<3; i++)
            {
				for(int j=0; j<3; j++)
                {
					this.I[i, j] += b.I[i, j];
				}
            }
		}
	}
}
