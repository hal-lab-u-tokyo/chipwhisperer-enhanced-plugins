/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPA.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  01-02-2025 09:07:43
*    Last Modified: 04-02-2025 04:15:10
*/


#ifndef SOCPA_H
#define SOCPA_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Arrays.hpp"
#include "AESLeakageModel.hpp"

namespace py = pybind11;

#define SQUARE(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))

#ifdef SOFT_QUAD_PRECISION
#include <quadfloat.h>
using QUADFLOAT = QuadFloat::QF128;
#else
using QUADFLOAT = long double;
#endif
using TRACE_T = double;
using RESULT_T = double;

class SOCPA {
public:
	const int NUM_GUESSES = 256;
	// Constructor
	SOCPA(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model) :
		byte_length(byte_length), num_points(num_points), total_traces(0), model(model), window_size(window_size) {
		// init arrays
		sum_trace = new TRACE_T[num_points]();
		sum_trace_square = new TRACE_T[num_points]();

		sum_hypothesis = new Array2D<int64_t>(byte_length, NUM_GUESSES);
		sum_hypothesis_square = new Array2D<int64_t>(byte_length, NUM_GUESSES);

		sum_hypothesis_trace = new Array3D<RESULT_T>(byte_length, NUM_GUESSES, num_points);

		sum_hypothesis_combined_trace = new Array4D<RESULT_T>(byte_length, NUM_GUESSES, num_points, window_size);

		sum_trace_x_win = new Array2D<TRACE_T>(num_points, window_size);
		sum_trace2_x_win = new Array2D<TRACE_T>(num_points, window_size);
		sum_trace_x_win2 = new Array2D<TRACE_T>(num_points, window_size);
		sum_trace2_x_win2 = new Array2D<TRACE_T>(num_points, window_size);


	};

	SOCPA(int byte_length, int num_points, AESLeakageModel::ModelBase *model) :
		SOCPA(byte_length, num_points, num_points, model) {};

	py::array_t<RESULT_T> calculate_correlation(py::array_t<TRACE_T> &py_traces,
								py::array_t<uint8_t> &py_plaintext,
								py::array_t<uint8_t> &py_ciphertext,
								py::array_t<uint8_t> &py_knownkey);

protected:
	int window_size;
	int byte_length;
	int num_traces;
	int num_points;
	int num_guesses;
	int total_traces;
	AESLeakageModel::ModelBase *model;

	Array2D<TRACE_T> *traces; // [0:num_traces][0:num_points]
	uint8_t *plaintext; // [0:num_traces][0:byte_length]
	uint8_t *ciphertext; // [0:num_traces][0:byte_length]
	uint8_t *knownkey; // [0:num_traces][0:byte_length

	// preserved intermediate results
	TRACE_T *sum_trace; // [0:num_points]
	TRACE_T *sum_trace_square; // [0:num_points]

	Array3D<int> *hypothetial_leakage; // [0:byte_length][0:num_guesses][0:num_traces]

	Array2D<int64_t> *sum_hypothesis; // [0:byte_length][0:num_guesses]
	Array2D<int64_t> *sum_hypothesis_square; // [0:byte_length][0:num_guesses]

	Array3D<RESULT_T> *sum_hypothesis_trace; // [0:byte_length][0:num_guesses][0:num_points][0:window_size]

	Array4D<RESULT_T> *sum_hypothesis_combined_trace; // [0:byte_length][0:num_guesses][0:num_points][0:window_size]

	// combined traces
	Array2D<TRACE_T> *sum_trace_x_win; // [0:num_points][0:window_size]
	Array2D<TRACE_T> *sum_trace2_x_win; // [0:num_points][0:window_size]
	Array2D<TRACE_T> *sum_trace_x_win2; // [0:num_points][0:window_size]
	Array2D<TRACE_T> *sum_trace2_x_win2; // [0:num_points][0:window_size]

	virtual void update_sum_trace();
	virtual void calculate_hypothesis();
	virtual void update_sum_hypothesis_trace();
	virtual void update_sum_hypothesis_combined_trace();
	virtual void calculate_correlation_subkey(Array4D<RESULT_T>* corr);


	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
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


#endif //SOCPA_H