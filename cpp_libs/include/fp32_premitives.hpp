/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/include/fp32_premitives.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  09-05-2025 18:44:29
*    Last Modified: 09-05-2025 18:44:31
*/

// for CUDA
#ifdef __CUDACC__
	#include <cuda.h>

	__device__ inline float2 quickTwoSum(float a, float b)
	{
		float s = a + b;
		float e = b - (s - a);
		return make_float2(s, e);
	}

	__device__ inline float4 twoSum(float2 a, float2 b)
	{
		float2 s = make_float2(a.x + b.x, a.y + b.y);
		float2 v = make_float2(s.x - a.x, s.y - a.y);
		float2 e = make_float2((a.x - (s.x - v.x)) + (b.x - v.x),
			(a.y - (s.y - v.y)) + (b.y - v.y));
		return make_float4(s.x, e.x, s.y, e.y);
	}

	__device__ inline float2 df64_add(float2 a, float2 b)
	{
		float4 st = twoSum(a, b);
		st.y += st.z;
		float2 xy = quickTwoSum(st.x, st.y);
		st.x = xy.x;
		st.y = xy.y;
		st.y += st.w;
		return quickTwoSum(st.x, st.y);
	}

	__device__ inline float2 twoProdFMA(float a, float b)
	{
		float p = a * b;
		float e = fma(a, b, -p); // a*b - p
		return make_float2(p, e);
	}

	__device__ inline float2 df64_mul(float2 a, float2 b)
	{
		float2 p = twoProdFMA(a.x, b.x);
		p.y += a.x * b.y;
		p.y += a.y * b.x;
		p = quickTwoSum(p.x, p.y);
		return p;
	}

	__device__ inline void df64_atomic_add(float2 *addr, float2 val)
	{
		float2 old;
		float2 desired;
		unsigned long long *uaddr = (unsigned long long *)addr;
		unsigned long long u_old;
		unsigned long long u_desired;
		do {
			old = *addr; // unsafe load
			desired = df64_add(old, val);
			u_old = *((unsigned long long*)(&old));
			u_desired = *((unsigned long long*)(&desired));
		} while (atomicCAS(uaddr, u_old, u_desired) != u_old);
	}
#endif