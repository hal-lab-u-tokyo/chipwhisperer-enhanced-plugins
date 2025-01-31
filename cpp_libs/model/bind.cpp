/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/model/bind.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:57:16
*    Last Modified: 30-01-2025 08:40:58
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "AESLeakageModel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(model_kernel, module) {
	module.doc() = "Model classes"; // optional module docstring

	py::class_<AESLeakageModel::ModelBase>(module, "ModelBase");

	py::class_<AESLeakageModel::SBoxOutput, AESLeakageModel::ModelBase>(module, "SBoxOutput")
		.def(py::init<>());

	py::class_<AESLeakageModel::SBoxInOutDiff, AESLeakageModel::ModelBase>(module, "SBoxInOutDiff")
		.def(py::init<>());

	py::class_<AESLeakageModel::PlaintextKeyXOR, AESLeakageModel::ModelBase>(module, "PlaintextKeyXOR")
		.def(py::init<>());

	py::class_<AESLeakageModel::PlaintextKeyXORDiff, AESLeakageModel::ModelBase>(module, "PlaintextKeyXORDiff")
		.def(py::init<>());

	py::class_<AESLeakageModel::LastRoundStateDiff, AESLeakageModel::ModelBase>(module, "LastRoundStateDiff")
		.def(py::init<>());

	py::class_<AESLeakageModel::LastRoundStateDiffAlternate, AESLeakageModel::ModelBase>(module, "LastRoundStateDiffAlternate")
		.def(py::init<>());

	py::class_<AESLeakageModel::LastRoundState, AESLeakageModel::ModelBase>(module, "LastRoundState")
		.def(py::init<>());

}
