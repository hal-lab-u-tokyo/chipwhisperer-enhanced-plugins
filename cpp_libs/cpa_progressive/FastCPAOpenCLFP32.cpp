#include "FastCPAOpenCLFP32.hpp"

#include <string>

using namespace std;

#define OCL_SUM_HYPOTHESIS_TRACE_FP32(...) #__VA_ARGS__
const char* FastCPAOpenCLFP32::sum_hypothesis_trace_kernel_code = 
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS_TRACE_FP32


FastCPAOpenCLFP32::FastCPAOpenCLFP32(int num_traces, int num_points, AESLeakageModel::ModelBase *model) : 
	FastCPAOpenCLBase(num_traces, num_points, model), traces_df64(nullptr), sum_hypothesis_trace_df64(nullptr)
{
	cl_int err;
	// create program
	sum_hypothesis_kernel_program =
		clCreateProgramWithSource(context, 1, get_sum_hypothesis_kernel_code(), nullptr, &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create program ("
							+ to_string(err) + ")");
	}
	sum_hypothesis_trace_kernel_program =
		clCreateProgramWithSource(context, 1, get_sum_hypothesis_trace_kernel_code(), nullptr, &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create program ("
							+ to_string(err) + ")");
	}

	// build program
	err = clBuildProgram(sum_hypothesis_kernel_program, 1, &device_id, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to build program \"sum_hypothesis_kernel\" ("
							+ to_string(err) + ")");
	}
	err = clBuildProgram(sum_hypothesis_trace_kernel_program, 1, &device_id, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to build program \"sum_hypothesis_trace_kernel\" ("
							+ to_string(err) + ")");
	}
	// create kernel
	sum_hypothesis_kernel = clCreateKernel(sum_hypothesis_kernel_program,
								"sum_hypothesis_kernel", &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create kernel \"sum_hypothesis_kernel\" ("
							+ to_string(err) + ")");
	}
	sum_hypothesis_trace_kernel
		= clCreateKernel(sum_hypothesis_trace_kernel_program,
								"sum_hypothesis_trace_kernel", &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create kernel \"sum_hypothesis_trace_kernel\" ("
							+ to_string(err) + ")");
	}

	clFinish(command_queue);

}

FastCPAOpenCLFP32::~FastCPAOpenCLFP32() {
	delete traces_df64;
	delete sum_hypothesis_trace_df64;
}



void FastCPAOpenCLFP32::setup_arrays(py::array_t<double> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey) {
	FastCPA::setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);

	cl_int err;
	// malloc gpu memory when not allocated
	if (cl_device_hypothetial_leakage == nullptr) {
		ALLOCATE_DEVICE_MEMORY(cl_device_hypothetial_leakage,
									CL_MEM_READ_ONLY,
									hypothetial_leakage->get_size());
	}
	if (cl_device_sum_hypothesis_trace == nullptr) {
		ALLOCATE_DEVICE_MEMORY(cl_device_sum_hypothesis_trace,
									CL_MEM_READ_WRITE,
									sum_hypothesis_trace->get_size());
		COPY_TO_DEVICE(cl_device_sum_hypothesis_trace,
						sum_hypothesis_trace->get_pointer(),
						sum_hypothesis_trace->get_size());
	}

	if (cl_device_traces == nullptr) {
		ALLOCATE_DEVICE_MEMORY(cl_device_traces,
									CL_MEM_READ_ONLY,
									traces->get_size());
	}
	// copy traces to GPU
	err = clFinish(command_queue);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to finish command queue ("
							+ to_string(err) + ")");
	}

	if (traces_df64 == nullptr) {
		traces_df64 = new Array2D<cl_float2>(num_traces, num_points);
	}
	if (sum_hypothesis_trace_df64 == nullptr) {
		sum_hypothesis_trace_df64 = new Array3D<cl_float2>(byte_length, NUM_GUESSES, num_points);
	}

	// convert double to float2
	double_to_float2(traces->get_pointer(), (cl_float2*)traces_df64->get_pointer(), num_traces * num_points);

	COPY_TO_DEVICE(cl_device_traces,
					traces_df64->get_pointer(),
					traces_df64->get_size());
	clFinish(command_queue);
}


void FastCPAOpenCLFP32::calculate_correlation_subkey(Array3D<double>* diff, long double *sumden2) {


	// offload to GPU for sum_hypothesis, sum_hypothesis_square
	run_sum_hypothesis_kernel();

	// offload to GPU for sum_hypothesis_trace
	run_sum_hypothesis_trace_kernel();


	// copy back

	COPY_FROM_DEVICE(cl_device_sum_hypothesis,
					(int64_t*)sum_hypothesis->get_pointer(),
					sum_hypothesis->get_size());
	COPY_FROM_DEVICE(cl_device_sum_hypothesis_square,
					(int64_t*)sum_hypothesis_square->get_pointer(),
					sum_hypothesis_square->get_size());
	COPY_FROM_DEVICE(cl_device_sum_hypothesis_trace,
					(cl_float2*)sum_hypothesis_trace_df64->get_pointer(),
					sum_hypothesis_trace->get_size());
	clFinish(command_queue);

	float2_to_double((cl_float2*)sum_hypothesis_trace_df64->get_pointer(), (double*)sum_hypothesis_trace->get_pointer(), byte_length * NUM_GUESSES * num_points);

	#ifdef _OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int guess = 0; guess < NUM_GUESSES; guess++) {
			// calc sumden1
			long double sumden1 = SQUARE(sum_hypothesis->at(byte_index, guess))
				- total_traces * sum_hypothesis_square->at(byte_index, guess);

			// calc sumnum
			long double sumnum;
			for (int p = 0; p < num_points; p++) {
				sumnum =
					total_traces * sum_hypothesis_trace->at(byte_index, guess, p)
					- sum_hypothesis->at(byte_index, guess) * sum_trace[p];
				diff->at(byte_index, guess, p) = sumnum / std::sqrt(sumden1 * sumden2[p]);
			}
		}
	}

}

void FastCPAOpenCLFP32::double_to_float2(const double *src, cl_float2 *dst, int length) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_float2(src[i]);
	}
}

void FastCPAOpenCLFP32::float2_to_double(const cl_float2 *src, double *dst, int length) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_double(src[i]);
	}
}