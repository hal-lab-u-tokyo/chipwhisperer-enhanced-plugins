/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/device_code.cl
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2024 12:30:39
*    Last Modified: 28-05-2025 02:34:18
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
		sum_hypothesis[byte_index * num_guess + guess] = sum_hyp;
		sum_hypothesis_square[byte_index * num_guess + guess] = sum_hyp_square;

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
			sum_hypothesis_trace[byte_index * num_guess * num_points + guess * num_points + point] = sum;
		}
	}
)
#undef OCL_SUM_HYPOTHESIS_TRACE

#ifndef OCL_SUM_TRACE
#define OCL_SUM_TRACE(...)
#endif

OCL_SUM_TRACE(
	__kernel void sum_trace_kernel(
		int num_traces, int num_points, int window_size,
		__global double *traces,
		__global double *sum_trace_x_win, __global double *sum_trace2_x_win,
		__global double *sum_trace_x_win2, __global double *sum_trace2_x_win2
	) {
		int point_index = get_global_id(0);
		int window_index = get_global_id(1);

		if (point_index < num_points && point_index + window_index + 1 < num_points && window_index < window_size) {
			double partial_sum_trace_x_win = 0;
			double partial_sum_trace2_x_win = 0;
			double partial_sum_trace_x_win2 = 0;
			double partial_sum_trace2_x_win2 = 0;
			for (int trace_index = 0; trace_index < num_traces; trace_index++) {
				double v1 = traces[trace_index * num_points + point_index];
				double v2 = traces[trace_index * num_points + point_index + window_index + 1];
				partial_sum_trace_x_win += v1 * v2;
				partial_sum_trace2_x_win += v1 * v1 * v2;
				partial_sum_trace_x_win2 += v1 * v2 * v2;
				partial_sum_trace2_x_win2 += v1 * v1 * v2 * v2;
			}
			sum_trace_x_win[point_index * window_size + window_index] = partial_sum_trace_x_win;
			sum_trace2_x_win[point_index * window_size + window_index] = partial_sum_trace2_x_win;
			sum_trace_x_win2[point_index * window_size + window_index] = partial_sum_trace_x_win2;
			sum_trace2_x_win2[point_index * window_size + window_index] = partial_sum_trace2_x_win2;
		}
	}
)
#undef OCL_SUM_TRACE


#ifndef OCL_SUM_HYPOTHESIS_COMBINED_TRACE
#define OCL_SUM_HYPOTHESIS_COMBINED_TRACE(...)
#endif
OCL_SUM_HYPOTHESIS_COMBINED_TRACE(

	inline void atomic_add_double(__global double *addr, double val) {
		ulong u_old;
		double desired;

		__global volatile ulong *uaddr = (__global volatile ulong *)addr;

		do {
			u_old = *uaddr; // unsafe load
			desired = as_double(u_old) + val;
		} while (atom_cmpxchg(uaddr, u_old, as_ulong(desired)) != u_old);
	}

	__kernel void sum_hypothesis_combined_trace_kernel(
		int num_traces, int start_point, int num_points, int window_size,
		int hyp_offset,
		__global int* hypothetial_leakage,
		__global double* traces,
		__global double* sum_hypothesis_combined_trace,
		__local double* trace_cache,
		__local double* hyp_cache
	) {
		const int point_per_block = get_local_size(2);
		const int window_per_block = get_local_size(1);
		const int trace_per_block = window_per_block; // assuming trace_per_block == window_per_block
		const int CACHE_DIM_Y = point_per_block + 1;

		const int point_offset = get_global_id(2);
		const int point_index = point_offset + start_point;
		const int trace_offset = get_group_id(0) * trace_per_block;
		const int end_trace = min(trace_per_block, num_traces - trace_offset);
		const int end_window = min(window_size, num_points - point_index - 1);

		if (point_index < num_points) {
			int p_lid = get_local_id(2);
			// copy trace data to shared memory
			int trace_index = trace_offset + get_local_id(1);
			if (trace_index < num_traces) {
				trace_cache[get_local_id(1) * CACHE_DIM_Y + p_lid] = traces[trace_index * num_points + point_index];
			}
			if (p_lid == 0) {
				hyp_cache[get_local_id(1)] = (trace_index < num_traces) ? (double)hypothetial_leakage[trace_index + hyp_offset] : 0;
			}
			// synchronize all threads in the work group
			barrier(CLK_LOCAL_MEM_FENCE);

			for (int w = get_local_id(1); w < end_window; w += window_per_block) {
				double sum = 0;
				for (int t = 0; t < end_trace; t++) {
					sum += hyp_cache[t] * trace_cache[t * CACHE_DIM_Y + p_lid] * traces[(trace_offset + t) * num_points + point_index + w + 1];
				}
				atomic_add_double(&sum_hypothesis_combined_trace[point_offset * window_size + w], sum);
			}
		}
	}
)
#undef OCL_SUM_HYPOTHESIS_COMBINED_TRACE

#ifndef OCL_SUM_HYPOTHESIS_COMBINED_TRACE_NOSM
#define OCL_SUM_HYPOTHESIS_COMBINED_TRACE_NOSM(...)
#endif

OCL_SUM_HYPOTHESIS_COMBINED_TRACE_NOSM(


	__kernel void sum_hypothesis_combined_trace_kernel(
		int num_traces, int start_point, int num_points, int window_size,
		int hyp_offset,
		__global int* hypothetial_leakage,
		__global double* traces,
		__global double* sum_hypothesis_combined_trace
	) {
		int point_index = get_global_id(0) + start_point;
		int window_index = get_global_id(1);

		double sum = 0;
		if (point_index < num_points && point_index + window_index + 1 < num_points && window_index < window_size) {
			for (int trace_index = 0; trace_index < num_traces; trace_index++) {
				int hyp = hypothetial_leakage[trace_index + hyp_offset];
				double v1 = traces[trace_index * num_points + point_index];
				double v2 = traces[trace_index * num_points + point_index + window_index + 1];
				sum += (double)hyp * v1 * v2;
			}
			// sum_hypothesis_combined_trace[point_index * window_size + window_index] += sum;
			int index = get_global_id(0) * window_size + window_index;
			sum_hypothesis_combined_trace[index] = sum;
		}
	}
)

#undef OCL_SUM_HYPOTHESIS_COMBINED_TRACE_NOSM


// ===================== FP32 emulatated implementation =====================

#ifndef OCL_FP32_PREMITIVES
#define OCL_FP32_PREMITIVES(...)
#endif

OCL_FP32_PREMITIVES(
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
)

#ifndef OCL_SUM_HYPOTHESIS_TRACE_FP32
#define OCL_SUM_HYPOTHESIS_TRACE_FP32(...)
#endif

OCL_SUM_HYPOTHESIS_TRACE_FP32(

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
			sum_hypothesis_trace[index] = sum;
		}
	}
)
#undef OCL_SUM_HYPOTHESIS_TRACE_FP32


#ifndef OCL_SUM_TRACE_FP32
#define OCL_SUM_TRACE_FP32(...)
#endif

OCL_SUM_TRACE_FP32(
	__kernel void sum_trace_kernel(
		int num_traces, int num_points, int window_size,
		__global float2 *traces,
		__global float2 *sum_trace_x_win, __global float2 *sum_trace2_x_win,
		__global float2 *sum_trace_x_win2, __global float2 *sum_trace2_x_win2
	) {
		int point_index = get_global_id(0);
		int window_index = get_global_id(1);

		if (point_index < num_points && point_index + window_index + 1 < num_points && window_index < window_size) {
			float2 partial_sum_trace_x_win = {0, 0};
			float2 partial_sum_trace2_x_win = {0, 0};
			float2 partial_sum_trace_x_win2 = {0, 0};
			float2 partial_sum_trace2_x_win2 = {0, 0};
			for (int trace_index = 0; trace_index < num_traces; trace_index++) {
				float2 v1 = traces[trace_index * num_points + point_index];
				float2 v2 = traces[trace_index * num_points + point_index + window_index + 1];
				partial_sum_trace_x_win = df64_add(partial_sum_trace_x_win, df64_mul(v1, v2));
				partial_sum_trace2_x_win = df64_add(partial_sum_trace2_x_win, df64_mul(df64_mul(v1, v1), v2));
				partial_sum_trace_x_win2 = df64_add(partial_sum_trace_x_win2, df64_mul(v1, df64_mul(v2, v2)));
				partial_sum_trace2_x_win2 = df64_add(partial_sum_trace2_x_win2, df64_mul(df64_mul(v1, v1), df64_mul(v2, v2)));
			}
			sum_trace_x_win[point_index * window_size + window_index] = partial_sum_trace_x_win;
			sum_trace2_x_win[point_index * window_size + window_index] = partial_sum_trace2_x_win;
			sum_trace_x_win2[point_index * window_size + window_index] = partial_sum_trace_x_win2;
			sum_trace2_x_win2[point_index * window_size + window_index] = partial_sum_trace2_x_win2;
		}
	}
)
#undef OCL_SUM_TRACE_FP32


#ifndef OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32
#define OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32(...)
#endif

OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32(

	inline void atomic_add_double(__global float2 *addr, float2 val) {
		ulong u_old;
		float2 desired;

		__global volatile ulong *uaddr = (__global volatile ulong *)addr;

		do {
			u_old = *uaddr; // unsafe load
			desired = df64_add(as_float2(u_old), val);
		} while (atom_cmpxchg(uaddr, u_old, as_ulong(desired)) != u_old);
	}

	__kernel void sum_hypothesis_combined_trace_kernel(
		int num_traces, int start_point, int num_points, int window_size,
		int hyp_offset,
		__global int* hypothetial_leakage,
		__global float2* traces,
		__global float2* sum_hypothesis_combined_trace,
		__local float2* trace_cache,
		__local float2* hyp_cache
	) {
		const int point_per_block = get_local_size(2);
		const int window_per_block = get_local_size(1);
		const int trace_per_block = window_per_block; // assuming trace_per_block == window_per_block
		const int CACHE_DIM_Y = point_per_block + 1;

		const int point_offset = get_global_id(2);
		const int point_index = point_offset + start_point;
		const int trace_offset = get_group_id(0) * trace_per_block;
		const int end_trace = min(trace_per_block, num_traces - trace_offset);
		const int end_window = min(window_size, num_points - point_index - 1);

		if (point_index < num_points) {
			int p_lid = get_local_id(2);
			// copy trace data to shared memory
			int trace_index = trace_offset + get_local_id(1);
			if (trace_index < num_traces) {
				trace_cache[get_local_id(1) * CACHE_DIM_Y + p_lid] = traces[trace_index * num_points + point_index];
			}
			if (p_lid == 0) {
				float hyp_float = (trace_index < num_traces) ? (float)hypothetial_leakage[trace_index + hyp_offset] : 0;
				hyp_cache[get_local_id(1)] = (float2)(hyp_float, 0);
			}
			// synchronize all threads in the work group
			barrier(CLK_LOCAL_MEM_FENCE);

			for (int w = get_local_id(1); w < end_window; w += window_per_block) {
				float2 sum = {0, 0};
				for (int t = 0; t < end_trace; t++) {
					// sum += hyp_cache[t] * trace_cache[t * CACHE_DIM_Y + p_lid] * traces[(trace_offset + t) * num_points + point_index + w + 1];
					float2 prod = df64_mul(hyp_cache[t], trace_cache[t * CACHE_DIM_Y + p_lid]);
					prod = df64_mul(prod, traces[(trace_offset + t) * num_points + point_index + w + 1]);
					sum = df64_add(sum, prod);
				}
				atomic_add_double(&sum_hypothesis_combined_trace[point_offset * window_size + w], sum);
			}
		}
	}
)
#undef OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32


#ifndef OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32_NOSM
#define OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32_NOSM(...)
#endif

OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32_NOSM(


	__kernel void sum_hypothesis_combined_trace_kernel(
		int num_traces, int start_point, int num_points, int window_size,
		int hyp_offset,
		__global int* hypothetial_leakage,
		__global float2* traces,
		__global float2* sum_hypothesis_combined_trace
	) {
		int point_index = get_global_id(0) + start_point;
		int window_index = get_global_id(1);

		float2 sum = {0, 0};
		if (point_index < num_points && point_index + window_index + 1 < num_points && window_index < window_size) {
			for (int trace_index = 0; trace_index < num_traces; trace_index++) {
				int hyp = hypothetial_leakage[trace_index + hyp_offset];
				float2 v1 = traces[trace_index * num_points + point_index];
				float2 v2 = traces[trace_index * num_points + point_index + window_index + 1];
				float2 prod = df64_mul((float2)((float)hyp, 0), v1);
				prod = df64_mul(prod, v2);
				sum = df64_add(sum, prod);
			}
			// sum_hypothesis_combined_trace[point_index * window_size + window_index] += sum;
			int index = get_global_id(0) * window_size + window_index;
			sum_hypothesis_combined_trace[index] = sum;
		}
	}
)

#undef OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32_NOSM