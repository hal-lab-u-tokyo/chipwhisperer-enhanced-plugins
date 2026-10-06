/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPACuda.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:56:58
*    Last Modified: 23-05-2025 12:09:43
*/

#ifndef FASTCPACUDA_H
#define FASTCPACUDA_H

#include "FastCPA.hpp"

#include <cuda_runtime.h>

#include <string>

#define CUDA_CHECK(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		std::string errMsg = "CUDA error at " + std::string(__FILE__) + ":" \
			+ std::to_string(__LINE__) + \
			" code=" + std::to_string(err) + "(" + \
			std::string(cudaGetErrorString(err)) + ") \"" + #call + "\""; \
		throw std::runtime_error(errMsg); \
	} \
}


class FastCPACudaBase : public FastCPA
{
public:
	FastCPACudaBase(int num_traces, int num_points, AESLeakageModel::ModelBase *model);
	~FastCPACudaBase();

protected:
	// device arrays
	double *device_traces;
	int *device_hypothetial_leakage;

	int64_t *device_sum_hypothesis;
	int64_t *device_sum_hypothesis_square;
	double *device_sum_hypothesis_trace;

	// overrided functions
	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey) = 0;
	virtual void calculate_hypothesis();
	virtual void calculate_correlation_subkey(Array3D<RESULT_T>* diff, QUADFLOAT *sumden2);
	virtual void calclualte_sumden2(QUADFLOAT *sumden2) {
		FastCPA::calclualte_sumden2(sumden2);
	};
	virtual void run_sum_hypothesis_trace_kernel() = 0;

};

class FastCPACuda : public FastCPACudaBase
{
public:
	FastCPACuda(int num_traces, int num_points, AESLeakageModel::ModelBase *model) :
				FastCPACudaBase(num_traces, num_points, model) {};
	~FastCPACuda() {};
protected:
	// overrided functions
	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);

	virtual void run_sum_hypothesis_trace_kernel();

};



#endif //FASTCPACUDA_H