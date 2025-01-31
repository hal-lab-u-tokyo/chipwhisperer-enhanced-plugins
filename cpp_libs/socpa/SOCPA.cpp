/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPA.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2025 06:33:19
*    Last Modified: 31-01-2025 19:52:36
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
#include <chrono>

namespace py = pybind11;
using namespace std;

py::array_t<RESULT_T> SOCPABase::calculate_correlation(py::array_t<TRACE_T> &py_traces,
											py::array_t<uint8_t> &py_plaintext,
											py::array_t<uint8_t> &py_ciphertext,
											py::array_t<uint8_t> &py_knownkey)
{

	setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);
	total_traces += num_traces;
	// auto end = chrono::system_clock::now();

	// update_sum_trace();

	// Array2D<QUADFLOAT> *sumden2 = new Array2D<QUADFLOAT>(num_points, window_size);
	// calclualte_sumden2(sumden2);

	calculate_hypothesis();

	py::array_t<RESULT_T> py_corr({byte_length, NUM_GUESSES, num_points, window_size});
	Array4D<RESULT_T>* corr = new Array4D<RESULT_T>((RESULT_T*)py_corr.request().ptr,
											byte_length, NUM_GUESSES, num_points, window_size);

	// calculate_correlation_subkey(corr, sumden2);
	// auto end2 = chrono::system_clock::now();

	// delete sumden2;

	printf("total traces: %d\n", total_traces);

	#pragma omp parallel for collapse(2)
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int p = 0; p < num_points; p++) {
			QUADFLOAT s1 = 0, s6 = 0;
			for (int t = 0; t < num_traces; t++) {
				s1 += traces->at(t, p); // reusable for each byte index, replaceable with sum_trace
				s6 += SQUARE(traces->at(t, p)); // reusable for each byte index, replaceable with sum_trace_square
			}

			int end_window = std::min(num_points, p + window_size + 1);
			for (int w = p + 1; w < end_window; w++) { 
				QUADFLOAT s2 = 0, s4 = 0, s8 = 0, s11 = 0, s12 = 0, s13 = 0; // reusable for each byte index
				for (int t = 0; t < num_traces; t++) {
					s2 += traces->at(t, w); // replaceable with sum_trace
					s8 += SQUARE(traces->at(t, w)); // replaceable with sum_trace_square
					s4 += traces->at(t, w) * traces->at(t, p); // combined
					s12 += SQUARE(traces->at(t, p)) * traces->at(t, w);
					s13 += SQUARE(traces->at(t, w)) * traces->at(t, p);
					s11 += SQUARE(traces->at(t, w)) * SQUARE(traces->at(t, p));
				}
				double n_lambda3 = (double)total_traces * s11 - 2.0 * (s2 * s12 + s1 * s13)  +
						(SQUARE(s2) * s6 + 4.0 * s1 * s2 * s4 + SQUARE(s1) * s8) / (double)total_traces -
						3.0 * SQUARE(s1 * s2) / (double)SQUARE(total_traces);
				double lambda2 = s4 - (s1 * s2)/(double)total_traces;
				if (byte_index == 0 && p == 0 ) {
					if ((n_lambda3 - lambda2) < 0) {
						printf("n_lambda3: %lf\n", (double)n_lambda3);
						printf("lambda2: %lf\n", (double)lambda2);
						printf("n_lambda3 - lambda2: %lf\n", (double)(n_lambda3 - lambda2));
					}
				}
				for (int guess = 0; guess < NUM_GUESSES; guess++) {
					int64_t s3 = 0, s5 = 0;
					QUADFLOAT s7 = 0, s9 = 0, s10 = 0;
					for (int t = 0; t < num_traces; t++) {
						auto hyp = hypothetial_leakage->at(byte_index, guess, t);
						s3 += hyp; // independent from points
						s9 += SQUARE(hyp);
						s5 += hyp * traces->at(t, p);
						s7 += hyp * traces->at(t, w);
						s10 += hyp * traces->at(t, p) * traces->at(t, w);
					}
					double n_lambda1 = (double)total_traces * s10 - (s1 * s7 + s2 * s5)  + (s1 * s2 * s3)/ (double)total_traces;
					corr->at(byte_index, guess, p, w - 1 - p) = (n_lambda1 - (RESULT_T)lambda2 * (RESULT_T)s3) /
						std::sqrt(((RESULT_T)n_lambda3 - SQUARE((RESULT_T)lambda2)) * (total_traces * (RESULT_T)s9 - SQUARE((RESULT_T)s3)));
						// check is nan
						// if (byte_index == 0 && guess == 0 && p == 0) {
						// 	printf("n_lambda1: %lf\n", (double)n_lambda1);
						// 	printf("lambda2: %lf\n", (double)lambda2);
						// 	printf("n_lambda3: %lf\n", (double)n_lambda3);
						// 	printf("s3: %lf\n", (double)s3);
						// 	printf("s9: %lf\n", (double)s9);
						// 	break;
						// }

				}
			}

		}
	}

	return py_corr;
}


void SOCPABase::setup_arrays(py::array_t<TRACE_T> &py_traces,
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

	combine();
};


void SOCPABase::update_sum_trace() {
	// update sum_trace and sum_trace_square
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int p = 0; p < num_points; p++) {
		for (int t = 0; t < num_traces; t++) {
			sum_trace[p] += traces->at(t, p);
			sum_trace_square[p] += SQUARE(traces->at(t, p));
		}
	}
}

void SOCPABase::calclualte_sumden2(Array2D<QUADFLOAT> *sumden2) {

// 	#ifdef _OPENMP
// 	#pragma omp parallel for
// 	#endif
// 	for (int p = 0; p < num_points; p++) {
// 		int end_window = std::min(window_size, num_points - p - 1);
// 		for (int win = 0; win < end_window; win++) {
// #ifdef SOFT_QUAD_PRECISION
// 			sumden2->at(p, win) = SQUARE(sum_trace->at(p, win)) - (QUADFLOAT)total_traces * sum_trace_square->at(p, win);
// #else
// 			sumden2->at(p, win) = std::fma(- (QUADFLOAT)total_traces, sum_trace_square->at(p, win), SQUARE(sum_trace->at(p, win)));
// #endif
// 		}
// 	}
}

void SOCPABase::calculate_hypothesis() {
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


void SOCPABase::calculate_correlation_subkey(Array4D<RESULT_T>* corr, Array2D<QUADFLOAT> *sumden2) {

	// QUADFLOAT sumden1;

	// auto start = chrono::system_clock::now();
	// // loop for each byte
	// #ifdef _OPENMP
	// #pragma omp parallel for private(sumden1) schedule(guided, 4)
	// #endif
	// for (int guess = 0; guess < NUM_GUESSES; guess++) {
	// 	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
	// 		for (int t = 0; t < num_traces; t++) {
	// 			auto hyp = hypothetial_leakage->at(byte_index, guess, t);
	// 			sum_hypothesis->at(byte_index, guess) += hyp;
	// 			sum_hypothesis_square->at(byte_index, guess) += SQUARE(hyp);
	// 			// sum up hypothesis * trace
	// 			for (int p = 0; p < num_points; p++) {
	// 				int end_window = std::min(window_size, num_points - p - 1);
	// 				for (int win = 0; win < end_window; win++) {
	// 					sum_hypothesis_trace->at(byte_index, guess, p, win)
	// 						+= hyp * combined_traces->at(t, p, win);
	// 				}
	// 			}
	// 		}

	// 		// calc sumden1
	// 		sumden1 = SQUARE(sum_hypothesis->at(byte_index, guess))
	// 		- total_traces * sum_hypothesis_square->at(byte_index, guess);

	// 		// calc sumnum
	// 		for (int p = 0; p < num_points; p++) {
	// 			int end_window = std::min(window_size, num_points - p - 1);
	// 			for (int win = 0; win < end_window; win++) {
	// 				QUADFLOAT sumnum = (QUADFLOAT)total_traces * sum_hypothesis_trace->at(byte_index, guess, p, win)
	// 				- sum_trace->at(p, win) * sum_hypothesis->at(byte_index, guess);

	// 				corr->at(byte_index, guess, p, win) = (RESULT_T)sumnum / std::sqrt((RESULT_T)sumden1 * (RESULT_T)sumden2->at(p, win));
	// 			}
	// 		}
	// 	}
	// }
	// auto end = chrono::system_clock::now();
	// auto dur = end - start;
	// std::cout << "calculation time: " << chrono::duration_cast<chrono::milliseconds>(dur).count() << "ms" << std::endl;

}



void ProductCombineSOCPA::combine() {

	// combined_traces = new Array3D<TRACE_T>(num_traces, num_points, window_size);

	// // combine traces
	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	// for (int t = 0; t < num_traces; t++) {
	// 	for (int p = 0; p < num_points; p++) {
	// 		int end_window = std::min(window_size, num_points - p - 1);
	// 		for (int w = 0; w < end_window; w++) {
	// 			combined_traces->at(t, p, w) = (traces->at(t, p) - average_trace[p]) *
	// 				(traces->at(t, p + w + 1) - average_trace[p + w + 1]);
	// 		}
	// 	}
	// }
}