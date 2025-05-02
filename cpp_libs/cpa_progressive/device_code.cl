/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/device_code.cl
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2024 12:30:39
*    Last Modified: 02-05-2025 09:50:28
*/

#ifndef OCL_SUM_HYPOTHESIS
#define OCL_SUM_HYPOTHESIS(...)
#endif

OCL_SUM_HYPOTHESIS(
	__kernel void sum_hypothesis_kernel(
		int byte_length, int num_guess, int num_traces,
		__global int* hypothetial_leakage,
		__global long* sum_hypothesis,
		__global long* sum_hypothesis_square)
	{
		int byte_index = get_global_id(0);
		int guess = get_global_id(1);

		long sum_hyp = 0;
		long sum_hyp_square = 0;
		for (int trace = 0; trace < num_traces; trace++) {
			int hyp = hypothetial_leakage[byte_index * num_guess * num_traces + guess * num_traces + trace];
			sum_hyp += hyp;
			sum_hyp_square += hyp * hyp;
		}
		sum_hypothesis[byte_index * num_guess + guess] += sum_hyp;
		sum_hypothesis_square[byte_index * num_guess + guess] += sum_hyp_square;

}
)
#undef OCL_SUM_HYPOTHESIS

#ifndef OCL_SUM_HYPOTHESIS_TRACE
#define OCL_SUM_HYPOTHESIS_TRACE(...)
#endif

OCL_SUM_HYPOTHESIS_TRACE(
	__kernel void sum_hypothesis_trace_kernel(
		int byte_length, int num_guess, int num_traces, int num_points,
		__global int* hypothetial_leakage,
		__global double* traces,
		__global double* sum_hypothesis_trace
	) {

		int byte_index = get_global_id(0);
		int guess = get_global_id(1);
		int point = get_global_id(2);

		if (point < num_points) {
			double sum = 0;
			for (int trace = 0; trace < num_traces; trace++) {
				int hyp = hypothetial_leakage[byte_index * num_guess * num_traces + guess * num_traces + trace];
				double trace_value = traces[trace * num_points + point];
				sum += hyp * trace_value;
			}
			sum_hypothesis_trace[byte_index * num_guess * num_points + guess * num_points + point] += sum;
		}
	}
)
#undef OCL_SUM_HYPOTHESIS_TRACE


#ifndef OCL_SUM_HYPOTHESIS_TRACE_FP32
#define OCL_SUM_HYPOTHESIS_TRACE_FP32(...)
#endif

OCL_SUM_HYPOTHESIS_TRACE_FP32(
	inline float2 quickTwoSum(float a, float b)
	{
		float s = a + b;
		float e = b - (s - a);
		return (float2)(s, e);
	}

	inline float4 twoSum(float2 a, float2 b)
	{
		float2 s = a + b;
		float2 v = s - a;
		float2 e = (a - (s - v)) + (b - v);
		return (float4)(s.x, e.x, s.y, e.y);
	}

	inline float2 df64_add(float2 a, float2 b)
	{
		float4 st = twoSum(a, b);
		st.y += st.z;
		st.xy = quickTwoSum(st.x, st.y);
		st.y += st.w;
		st.xy = quickTwoSum(st.x, st.y);
		return st.xy;
	}

	inline float2 twoProdFMA(float a, float b)
	{
		float p = a * b;
		float e = fma(a, b, -p); // a*b - p
		return (float2)(p, e);
	}

	inline float2 df64_mul(float2 a, float2 b)
	{
		float2 p = twoProdFMA(a.x, b.x);
		p.y += a.x * b.y;
		p.y += a.y * b.x;
		p = quickTwoSum(p.x, p.y);
		return p;
	}
	__kernel void sum_hypothesis_trace_kernel(
		int byte_length, int num_guess, int num_traces, int num_points,
		__global int* hypothetial_leakage,
		__global float2* traces,
		__global float2* sum_hypothesis_trace
	) {
		int byte_index = get_global_id(0);
		int guess = get_global_id(1);
		int point = get_global_id(2);

		if (point < num_points) {
			float2 sum = { 0, 0 };
			for (int trace = 0; trace < num_traces; trace++) {
				int hyp = hypothetial_leakage[byte_index * num_guess * num_traces + guess * num_traces + trace];
				float2 trace_value = traces[trace * num_points + point];
				float2 prod = df64_mul((float2)((float)hyp, 0), trace_value);
				sum = df64_add(sum, prod);
			}
			// sum_hypothesis_trace[byte_index * num_guess * num_points + guess * num_points + point] += sum;
			int index = byte_index * num_guess * num_points + guess * num_points + point;
			sum_hypothesis_trace[index] = df64_add(sum_hypothesis_trace[index], sum);
		}
	}
)
#undef OCL_SUM_HYPOTHESIS_TRACE_FP32

