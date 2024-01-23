/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/bind.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:57:16
*    Last Modified: 23-01-2024 16:57:17
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "FastCPA.hpp"

#include "AESLeakageModel.hpp"

PYBIND11_MODULE(cpa_kernel, module) {
	module.doc() = "pybind11 example plugin"; // optional module docstring

	py::class_<FastCPA>(module, "FastCPA")
		.def(py::init<int, int, AESLeakageModel::ModelBase*>())
		.def("calculate_correlation", &FastCPA::calculate_correlation);


	py::class_<AESLeakageModel::ModelBase>(module, "ModelBase");

	py::class_<AESLeakageModel::SBoxOutput, AESLeakageModel::ModelBase>(module, "SBoxOutput")
		.def(py::init<>());

	py::class_<AESLeakageModel::SBoxInOutDiff, AESLeakageModel::ModelBase>(module, "SBoxInOutDiff")
		.def(py::init<>());

	py::class_<AESLeakageModel::LastRoundStateDiff, AESLeakageModel::ModelBase>(module, "LastRoundStateDiff")
		.def(py::init<>());
}
