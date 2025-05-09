/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPACudaFP32.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  09-05-2025 18:43:31
*    Last Modified: 09-05-2025 18:43:32
*/



#ifndef SOCPACUDAFP32_H
#define SOCPACUDAFP32_H

#include "SOCPACuda.hpp"

class SOCPACudaFP32 : public SOCPACuda
{
public:
	SOCPACudaFP32(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model, bool use_shared_mem);
	~SOCPACudaFP32();

protected:
	virtual void calculate_sum_hypothesis_trace();
	virtual void calculate_sum_trace();

	virtual void run_sum_hypothesis_coumbined_trace_kernel(int start_point, int hyp_offset);
	virtual void run_sum_hypothesis_coumbined_trace_kernel_nosm(int start_point, int hyp_offset);

	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);

	// for format conversion
	Array2D<float2> *traces_df64;
	Array3D<float2> *sum_hypothesis_trace_df64;
	Array2D <float2> *sum_trace_x_win_df64, *sum_trace2_x_win_df64;
	Array2D <float2> *sum_trace_x_win2_df64, *sum_trace2_x_win2_df64;

	float2 *sum_hypothesis_combined_trace_df64;

private:
	void double_to_float2(const double *src, float2 *dst, int length);
	void float2_to_double(const float2 *src, double *dst, int length);

};

#endif //SOCPACUDAFP32_H