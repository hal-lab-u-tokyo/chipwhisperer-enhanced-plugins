/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPACuda.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  01-02-2025 09:11:42
*    Last Modified: 01-02-2025 09:17:49
*/

#ifndef SOCPACUDA_H
#define SOCPACUDA_H

#include "SOCPA.hpp"

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


class SOCPACuda : public SOCPA
{
public:
	// Constructor
	SOCPACuda(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model);
	~SOCPACuda();


protected:

	// device arrays
	double *device_traces;
	int *device_hypothetial_leakage;

	int64_t *device_sum_hypothesis;
	int64_t *device_sum_hypothesis_square;
	double *device_sum_hypothesis_trace;

	double *device_sum_hypothesis_combined_trace;
	Array4D<double> *dbg_sum_hypothesis_combined_trace;

	virtual void update_sum_hypothesis_trace();
	virtual void update_sum_hypothesis_combined_trace();
	// virtual void update_sum_trace();
	virtual void calculate_hypothesis();
	// virtual void update_sum_hypothesis_trace();
	// virtual void update_sum_hypothesis_combined_trace();
	// virtual void calculate_correlation_subkey(Array4D<RESULT_T>* corr);


	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);


};


#endif //SOCPACUDA_H