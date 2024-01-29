/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPA.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:57:38
*    Last Modified: 29-01-2024 22:47:46
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "FastCPA.hpp"

#include "AESLeakageModel.hpp"

// check availability of OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <chrono>

namespace py = pybind11;
using namespace std;

py::array_t<double> FastCPA::calculate_correlation(py::array_t<double> &py_traces,
											py::array_t<uint8_t> &py_plaintext,
											py::array_t<uint8_t> &py_ciphertext,
											py::array_t<uint8_t> &py_knownkey)
{
	setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);
	total_traces += num_traces;

	update_sum_trace();

	QUADFLOAT *sumden2 = new QUADFLOAT[num_points];
	calclualte_sumden2(sumden2);

	calculate_hypothesis();

	py::array_t<double> py_diff({byte_length, NUM_GUESSES, num_points});
	Array3D<double>* diff = new Array3D<double>((double*)py_diff.request().ptr,
											byte_length, NUM_GUESSES, num_points);
	calculate_correlation_subkey(diff, sumden2);

	delete[] sumden2;

	return py_diff;
}

void FastCPA::setup_arrays(py::array_t<double> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey) {

	const auto &buf_info = py_traces.request();
	// check shape
	if (buf_info.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}
	num_traces = buf_info.shape[0];
	if (buf_info.shape[1] != num_points) {
		throw std::runtime_error("each trace length must be the same as num_points");
	}

	// get pointer
	traces = new Array2D<double>((double*)py_traces.request().ptr,
										num_traces, num_points);

	// verify shape
	verify_array_size(py_plaintext, num_traces, byte_length, "plaintext");
	verify_array_size(py_ciphertext, num_traces, byte_length, "ciphertext");
	verify_array_size(py_knownkey, num_traces, byte_length, "knownkey");

	plaintext = (uint8_t*)py_plaintext.request().ptr;
	ciphertext = (uint8_t*)py_ciphertext.request().ptr;
	knownkey = (uint8_t*)py_knownkey.request().ptr;

	hypothetial_leakage = new Array3D<int>(byte_length, NUM_GUESSES, num_traces);
};


void FastCPA::update_sum_trace() {
	// update sum_trace and sum_trace_square
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int p = 0; p < num_points; p++) {
		for (int t = 0; t < num_traces; t++) {
			QUADFLOAT prev = sum_trace[p];
			sum_trace[p] += traces->at(t, p);
			sum_trace_square[p] += SQUARE(traces->at(t, p));
		}
	}
}

void FastCPA::calclualte_sumden2(QUADFLOAT *sumden2) {

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int p = 0; p < num_points; p++) {
		sumden2[p] = SQUARE(sum_trace[p]) - (QUADFLOAT)total_traces * sum_trace_square[p];
	}
}

void FastCPA::calculate_hypothesis() {
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	// loop for each byte
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		// loop for each guess
		for (int guess = 0; guess < NUM_GUESSES; guess++) {
			// fill the guess key
			uint8_t *key = new uint8_t[byte_length]();
			key[byte_index] = guess;
			for (int t = 0; t < num_traces; t++) {
				// calc hypothetical leakage using the model
				hypothetial_leakage->at(byte_index, guess, t) =
					model->leakage(plaintext + t * byte_length,
									ciphertext + t * byte_length,
									key,
									byte_index);
			}
			delete[] key;
		}
	}
}


void FastCPA::calculate_correlation_subkey(Array3D<double>* diff, QUADFLOAT *sumden2) {

	QUADFLOAT sumden1;
	// loop for each byte
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2) private(sumden1)
	#endif
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int guess = 0; guess < NUM_GUESSES; guess++) {
			for (int t = 0; t < num_traces; t++) {
				auto hyp = hypothetial_leakage->at(byte_index, guess, t);
				sum_hypothesis->at(byte_index, guess) += hyp;
				sum_hypothesis_square->at(byte_index, guess) += SQUARE(hyp);
				// sum up hypothesis * trace
				for (int p = 0; p < num_points; p++) {
					sum_hypothesis_trace->at(byte_index, guess, p)
						+= hyp * traces->at(t, p);
				}
			}

			// calc sumden1
			sumden1 = SQUARE(sum_hypothesis->at(byte_index, guess))
			- total_traces * sum_hypothesis_square->at(byte_index, guess);

			// calc sumnum
			for (int p = 0; p < num_points; p++) {
				QUADFLOAT sumnum = (QUADFLOAT)total_traces * sum_hypothesis_trace->at(byte_index, guess, p)
					- sum_trace[p] * sum_hypothesis->at(byte_index, guess);

				diff->at(byte_index, guess, p) = (double)sumnum / std::sqrt((double)sumden1 * (double)sumden2[p]);
			}
		}
	}

}

