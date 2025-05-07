/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPA.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2025 06:33:19
*    Last Modified: 07-05-2025 01:05:43
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "SOCPA.hpp"
#include "AESLeakageModel.hpp"

// check availability of OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <algorithm>

namespace py = pybind11;
using namespace std;

py::array_t<RESULT_T> SOCPA::calculate_correlation(py::array_t<TRACE_T> &py_traces,
											py::array_t<uint8_t> &py_plaintext,
											py::array_t<uint8_t> &py_ciphertext,
											py::array_t<uint8_t> &py_knownkey)
{

	setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);

	// sum up each sample of traces
	calculate_sum_trace();

	// calculate hypothetical leakage from the leakage model
	calculate_hypothesis();

	// sum up the product of hypothetical leakage and traces
	calculate_sum_hypothesis_trace();


	py::array_t<RESULT_T> py_corr({byte_length, NUM_GUESSES, num_points});


	Array3D<RESULT_T>* corr = new Array3D<RESULT_T>((RESULT_T*)py_corr.request().ptr,
											byte_length, NUM_GUESSES, num_points);

	// calculate correlation
	calculate_correlation_subkey(corr);

	return py_corr;
}


void SOCPA::setup_arrays(py::array_t<TRACE_T> &py_traces,
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
	traces = new Array2D<TRACE_T>((TRACE_T*)py_traces.request().ptr,
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


void SOCPA::calculate_sum_trace() {
	// sum up each sample of traces
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int p = 0; p < num_points; p++) {
		for (int t = 0; t < num_traces; t++) {
			auto v1 = traces->at(t, p);
			sum_trace[p] += v1;
			sum_trace_square[p] += SQUARE(v1);
			int end_window = std::min(num_points, p + window_size + 1);
			for (int w = p + 1; w < end_window; w++) {
				auto v2 = traces->at(t, w);
				sum_trace_x_win->at(p, w - p - 1) += v1 * v2;
				sum_trace2_x_win->at(p, w - p - 1) += SQUARE(v1) * v2;
				sum_trace_x_win2->at(p, w - p - 1) += v1 * SQUARE(v2);
				sum_trace2_x_win2->at(p, w - p - 1) += SQUARE(v1) * SQUARE(v2);
			}
		}
	}

}

void SOCPA::calculate_sum_hypothesis_trace() {
	// sum up the product of hypothetical leakage and traces
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int guess = 0; guess < NUM_GUESSES; guess++) {
			for (int t = 0; t < num_traces; t++) {
				auto hyp = hypothetial_leakage->at(byte_index, guess, t);
				sum_hypothesis->at(byte_index, guess) += hyp;
				sum_hypothesis_square->at(byte_index, guess) += SQUARE(hyp);
				for (int p = 0; p < num_points; p++) {
					sum_hypothesis_trace->at(byte_index, guess, p)
							+= hyp * traces->at(t, p);
				}
			}
		}
	}
}

void SOCPA::calculate_hypothesis() {
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


void SOCPA::calculate_correlation_subkey(Array3D<RESULT_T>* corr) {

	/* calculation formula and symbols (s1-s13) are the same as the following paper:
		Bottinelli, Paul, and Joppe W. Bos. "Computational aspects of correlation power analysis." Journal of Cryptographic Engineering 7 (2017): 167-181.
	*
	* */

	// tiling parameter
	const int tile_point = this->point_tile_size;
	const int tile_trace = this->trace_tile_size;

	QUADFLOAT div_n = (QUADFLOAT)(1.0/(double)num_traces);
	QUADFLOAT div_nn = SQUARE(div_n);

	#ifdef _OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {

		for (int guess = 0; guess < NUM_GUESSES; guess++) {

			auto s3 = sum_hypothesis->at(byte_index, guess);
			auto s9 = sum_hypothesis_square->at(byte_index, guess);

			// temporary storage
			auto sum_hypothesis_combined_trace = new RESULT_T[tile_point * window_size]();

			for (int tp = 0; tp < num_points; tp += tile_point) {

				for (int tt = 0; tt < num_traces; tt += tile_trace) {

					// sum up the product of hypothetical leakage and combined points
					for (int pp = 0; pp < std::min(tile_point, num_points - tp); pp++) {
						auto p = pp + tp;
						int end_window = std::min(window_size, num_points -  p - 1);

						for (int t = 0; t < std::min(tile_trace, num_traces - tt); t++) {
							auto hyp = hypothetial_leakage->at(byte_index, guess, t + tt);
							auto v1 = traces->at(t + tt, p);
							for (int w = 0; w < end_window; w++) {
								sum_hypothesis_combined_trace[pp * window_size + w]
								 += hyp * v1 * traces->at(t + tt, p + w + 1);
							}
						} // end partial trace

					} // end partial point

				} // end trace tile

				// calculate the correlation
				for (int pp = 0; pp < std::min(tile_point, num_points - tp); pp++) {
					auto p = pp + tp;
					int end_window = std::min(window_size, num_points - p - 1);

					auto s1 = (QUADFLOAT)sum_trace[p];
					auto s5 = sum_hypothesis_trace->at(byte_index, guess, p);
					auto s6 = (QUADFLOAT)sum_trace_square[p];

					RESULT_T max_corr = 0.0;
					int max_win = 0;

					// find max correlation
					for (int w = 0; w < end_window; w++) {
						auto s2 = (QUADFLOAT)sum_trace[p + w + 1];
						auto s4 = (QUADFLOAT)sum_trace_x_win->at(p, w);
						auto s7 = sum_hypothesis_trace->at(byte_index, guess, p + w + 1);
						auto s8 = (QUADFLOAT)sum_trace_square[p + w + 1];
						auto s10 = sum_hypothesis_combined_trace[pp * window_size + w];
						auto s11 = (QUADFLOAT)sum_trace2_x_win2->at(p, w);
						auto s12 = (QUADFLOAT)sum_trace2_x_win->at(p, w);
						auto s13 = (QUADFLOAT)sum_trace_x_win2->at(p, w);

						QUADFLOAT n_lambda3 = (QUADFLOAT)num_traces * s11 -
									QUADFLOAT(2.0) * (s2 * s12 + s1 * s13)  +
									(SQUARE(s2) * s6 + QUADFLOAT(4.0) * s1 * s2 * s4 + SQUARE(s1) * s8) * div_n -
									QUADFLOAT(3.0) * SQUARE(s1 * s2) * div_nn;
						QUADFLOAT lambda2 = s4 - (s1 * s2) * div_n;
						QUADFLOAT n_lambda1 = (QUADFLOAT)num_traces * s10 - (s1 * s7 + s2 * s5) + (s1 * s2 * s3) * div_n;
						RESULT_T corr = (RESULT_T)(n_lambda1 - lambda2 * s3) /
										std::sqrt((RESULT_T)(((n_lambda3 - SQUARE(lambda2)) * (num_traces * s9 - SQUARE(s3)))));

						// check if the correlation is maximum
						if (std::abs(corr) > std::abs(max_corr)) {
							max_corr = corr;
							max_win = w;
						}
					} // end window

					// store the result
					corr->at(byte_index, guess, p) = max_corr;
					max_combined_offset->at(byte_index, guess, p) = max_win;

				} // end partial point

				// clear the temporary storage

				std::fill(sum_hypothesis_combined_trace, sum_hypothesis_combined_trace + tile_point * window_size, 0.0);
			} // end point tile

			delete[] sum_hypothesis_combined_trace;
		} // end guess

	} // end byte

}