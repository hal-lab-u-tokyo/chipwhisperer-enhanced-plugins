/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPA.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:56:54
*    Last Modified: 23-01-2024 16:56:55
*/

#ifndef FAST_CPA_H
#define FAST_CPA_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "Arrays.hpp"
#include "AESLeakageModel.hpp"

namespace py = pybind11;

#define SQUARE(x) ((x) * (x))

class FastCPA {
public:
	static constexpr int NUM_GUESSES = 256;
	// Constructor
	FastCPA(int byte_length, int num_points, AESLeakageModel::ModelBase *model) :
		byte_length(byte_length), num_points(num_points), total_traces(0), model(model) {

		// init arrays
		sum_trace = new long double[num_points]();
		sum_trace_square = new long double[num_points]();

		sum_hypothesis = new Array2D<int64_t>(byte_length, NUM_GUESSES);
		sum_hypothesis_square = new Array2D<int64_t>(byte_length, NUM_GUESSES);

		sum_hypothesis_trace = new Array3D<double>(byte_length, NUM_GUESSES, num_points);

	};

	py::array_t<double> calculate_correlation(py::array_t<double> &py_traces,
								py::array_t<uint8_t> &py_plaintext,
								py::array_t<uint8_t> &py_ciphertext,
								py::array_t<uint8_t> &py_knownkey);

protected:
	int byte_length;
	int num_traces;
	int num_points;
	int num_guesses;
	int total_traces;
	AESLeakageModel::ModelBase *model;

	Array2D<double> *traces; // [0:num_traces][0:num_points]
	uint8_t *plaintext; // [0:num_traces][0:byte_length]
	uint8_t *ciphertext; // [0:num_traces][0:byte_length]
	uint8_t *knownkey; // [0:num_traces][0:byte_length

	// preserved intermediate results
	long double *sum_trace; // [0:num_points]
	long double *sum_trace_square; // [0:num_points]

	Array3D<int> *hypothetial_leakage; // [0:byte_length][0:num_guesses][0:num_traces]

	Array2D<int64_t> *sum_hypothesis; // [0:byte_length][0:num_guesses]
	Array2D<int64_t> *sum_hypothesis_square; // [0:byte_length][0:num_guesses]

	Array3D<double> *sum_hypothesis_trace; // [0:byte_length][0:num_guesses][0:num_points]

	virtual void update_sum_trace();
	virtual void calclualte_sumden2(long double *sumden2);
	virtual void calculate_hypothesis();
	virtual void calculate_correlation_subkey(Array3D<double>* diff, long double *sumden2);


	virtual void setup_arrays(py::array_t<double> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);

	virtual void release_arrays() {};

	template <typename ARRAY_T>
	void verify_array_size(py::array_t<ARRAY_T> &py_array, int dim_x,
							 int dim_y, std::string name) {
		const auto &buf_info = py_array.request();
		if (buf_info.ndim != 2) {
			throw std::runtime_error(name + ": Dimensions must be two");
		}
		if (buf_info.shape[0] != num_traces) {
			throw std::runtime_error(name + " must have the same length as traces");
		}
		if (buf_info.shape[1] != byte_length) {
			throw std::runtime_error("each array of " + name + " length must be the same as byte_length");
		}
	}
};

#endif //FAST_CPA_H