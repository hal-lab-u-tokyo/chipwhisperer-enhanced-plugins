/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPACudaFP32.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-05-2025 12:13:38
*    Last Modified: 23-05-2025 18:51:34
*/

#ifndef FASTCPACUDAFP32_H
#define FASTCPACUDAFP32_H

#include "FastCPACuda.hpp"


class FastCPACudaFP32 : public FastCPACudaBase
{
public:
	FastCPACudaFP32(int num_traces, int num_points, AESLeakageModel::ModelBase *model) :
		FastCPACudaBase(num_traces, num_points, model),
		traces_df64(nullptr), sum_hypothesis_trace_df64(nullptr) {};

	~FastCPACudaFP32();

protected:

	// overrided functions
	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);
	virtual void run_sum_hypothesis_trace_kernel();

private:
	Array2D<float2> *traces_df64;
	Array3D<float2> *sum_hypothesis_trace_df64;


	void double_to_float2(const double *src, float2 *dst, int length);
	void float2_to_double(const float2 *src, double *dst, int length);


};

#endif //FASTCPACUDAFP32_H