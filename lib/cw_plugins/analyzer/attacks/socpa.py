###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /lib/cw_plugins/analyzer/attacks/socpa.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  01-02-2025 09:07:18
#   Last Modified: 20-06-2025 16:03:56
###

from chipwhisperer.common.utils.parameter import setupSetParam
from chipwhisperer.analyzer.attacks._base import AttackBaseClass
from chipwhisperer.analyzer.attacks.algorithmsbase import AlgorithmsBase
from chipwhisperer.analyzer.attacks.models.base import ModelsBase
import chipwhisperer as cw
from warnings import warn

import numpy as np

from .cpa_algorithms.models import get_c_model
from .socpa_stats import SOCPAResults

class SOCPAAlgorithm(AlgorithmsBase):
    """
    Second Order CPA Attack
    """

    def __init__(self):
        super().__init__()
        self._window_size = 1
        self.point_tile_size = None
        self.trace_tile_size = None

    def set_window_size(self, winsize):
        self._window_size = winsize

    def get_window_size(self):
        return self._window_size

    def set_reporting_interval(self, ri):
        warn("Reporting interval is not used in SOCPA")

    def setModel(self, model):
        self.model = model
        if model:
            self.stats = SOCPAResults(model.getNumSubKeys(), model.getPermPerSubkey(), self._window_size)

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        return SOCPA(byte_len, numpoints, self._window_size, model)

    def set_point_tile_size(self, tile_size):
        self.point_tile_size = tile_size


    def set_trace_tile_size(self, tile_size):
        self.trace_tile_size = tile_size

    def addTraces(self, traceSource, tracerange, progressBar=None, pointRange=None):

        numtraces = tracerange[1] - tracerange[0]
        numpoints = pointRange[1] - pointRange[0]


        if self._window_size > numpoints:
            raise ValueError("Window size cannot be greater than number of points in trace")

        byte_len = max(self.brange) + 1

        model = get_c_model(self.model)

        socpa = self.getSoCpaKernel(byte_len, numpoints, model)

        if self.point_tile_size is not None:
            point_tile_size = min(self.point_tile_size, numpoints)
            socpa.set_point_tile_size(point_tile_size)

        if self.trace_tile_size is not None:
            trace_tile_size = min(self.trace_tile_size, numtraces)
            socpa.set_trace_tile_size(trace_tile_size)

        trange = range(0, numtraces)
        part_trace = np.array([traceSource.get_trace(t + tracerange[0])[pointRange[0]:pointRange[1]] for t in trange])
        part_textin = np.array([traceSource.get_textin(t + tracerange[0]) for t in trange])
        if type(traceSource.get_textout(0)) == bytes:
            part_textout = np.array([np.frombuffer(traceSource.get_textout(t + tracerange[0]), dtype=np.uint8) for t in trange])
        else:
            part_textout = np.array([traceSource.get_textout(t + tracerange[0]) for t in trange])

        part_knownkey = np.array([traceSource.get_known_key(t + tracerange[0]) for t in trange])

        # register known key
        if part_knownkey[0] is not None:
            self.stats.set_known_key(self.process_known_key(part_knownkey[0]))

        # run c++ library
        corr = socpa.calculate_correlation(part_trace, part_textin, part_textout, part_knownkey)
        max_conb_offset = socpa.get_max_combined_offset()

        # save statistics
        for bnum in self.brange:
            self.stats.store_correlation(bnum, corr[bnum], max_conb_offset[bnum])

        del part_trace, part_textin, part_textout, part_knownkey

        # Run callback
        if self.sr:
            self.sr()

        # del socpa
        del socpa

class SOCPAAlgorithmCuda(SOCPAAlgorithm):
    """
    Second Order CPA Attack with CUDA
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_cuda_kernel import SOCPACuda
        return SOCPACuda(byte_len, numpoints, self._window_size, model, True)

class SOCPAAlgorithmCudaNoSM(SOCPAAlgorithm):
    """
    Second Order CPA Attack with CUDA (No shared memory)
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_cuda_kernel import SOCPACuda
        return SOCPACuda(byte_len, numpoints, self._window_size, model, False)

class SOCPAAlgorithmCudaFP32(SOCPAAlgorithm):
    """
    Second Order CPA Attack with CUDA (FP32 emulation)
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_cuda_kernel import SOCPACudaFP32
        return SOCPACudaFP32(byte_len, numpoints, self._window_size, model, True)
    
class SOCPAAlgorithmCudaFP32NoSM(SOCPAAlgorithm):
    """
    Second Order CPA Attack with CUDA (FP32 emulation, No shared memory)
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_cuda_kernel import SOCPACudaFP32
        return SOCPACudaFP32(byte_len, numpoints, self._window_size, model, False)

class SOCPAAlgorithmOpenCL(SOCPAAlgorithm):
    """
    Second Order CPA Attack with OpenCL
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_opencl_kernel import SOCPAOpenCL
        return SOCPAOpenCL(byte_len, numpoints, self._window_size, model, True)

class SOCPAAlgorithmOpenCLNoSM(SOCPAAlgorithm):
    """
    Second Order CPA Attack with OpenCL (No shared memory)
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_opencl_kernel import SOCPAOpenCL
        return SOCPAOpenCL(byte_len, numpoints, self._window_size, model, False)

class SOCPAAlgorithmOpenCLFP32(SOCPAAlgorithm):
    """
    Second Order CPA Attack with OpenCL (FP32 emulation)
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_opencl_kernel import SOCPAOpenCLFP32
        return SOCPAOpenCLFP32(byte_len, numpoints, self._window_size, model, True)

class SOCPAAlgorithmOpenCLFP32NoSM(SOCPAAlgorithm):
    """
    Second Order CPA Attack with OpenCL (FP32 emulation, No shared memory)
    """

    def getSoCpaKernel(self, byte_len, numpoints, model):
        from .cpa_algorithms.socpa_kernel import SOCPA
        from .cpa_algorithms.socpa_opencl_kernel import SOCPAOpenCLFP32
        return SOCPAOpenCLFP32(byte_len, numpoints, self._window_size, model, False)

class SOCPA(AttackBaseClass):
    """Second Order CPA Attack"""

    def __init__(self, proj, leak_model : ModelsBase, algorithm : SOCPAAlgorithm):
        self._analysisAlgorithm = algorithm()
        super().__init__()
        self.updateScript()
        # below will set self.attack = algorithm, self.attackModel = leak_model
        self.set_analysis_algorithm(algorithm, leak_model)
        self._traceEnd = 0
        self.change_project(proj)


    def change_project(self, proj):
        self.set_trace_source(proj.trace_manager())
        self.setProject(proj)
        self.set_target_subkeys(range(len(proj.keys[0])))
        self.set_point_range((0, len(proj.waves[0])))
        self.set_trace_range(0, len(proj.traces)-1)

    def set_trace_end(self, tnum):
        self._traceEnd = min(len(self.project().traces)-1, tnum)
        if self._traceEnd != tnum:
            print("Warning: Trace number out of range. Setting to last trace.")

    def get_trace_end(self):
        return self._traceEnd

    def set_trace_range(self, start, end):
        if end <= start:
            raise ValueError("End trace cannot be before start trace")
        self.set_trace_start(start)
        self.set_trace_end(end)

    trace_range = property(fset=lambda self, trange: self.set_trace_range(trange[0], trange[1]))

    point_range = property(fset=lambda self, prange: self.set_point_range(prange))

    def get_trace_range(self):
        return (self.get_trace_start(), self.get_trace_end())

    def set_trace_tile_size(self, tile_size):
        self.attack.set_trace_tile_size(tile_size)

    def set_point_tile_size(self, tile_size):
        self.attack.set_point_tile_size(tile_size)

    @property
    def window_size(self):
        return self.attack.get_window_size()

    @window_size.setter
    def window_size(self, winsize):
        self.attack.set_window_size(winsize)


    # report interval is not used in SOCPA but we keep it for compatibility with jupyter notebooks callbacks
    @property
    def reporting_interval(self):
        return self.get_trace_end() - self.get_trace_start() + 1

    def process_known_key(self, inpkey):
        if inpkey is None:
            return None

        if hasattr(self.attack, 'process_known_key'):
            return self.attack.process_known_key(inpkey)
        else:
            return inpkey

    def getStatistics(self):
        return self.attack.getStatistics()

    @property
    def results(self):
        return self.getStatistics()

    def process_traces(self, callback=None):

        self.attack.setModel(self.attackModel)
        self.attack.get_statistics().clear()

        self.attack.setTargetSubkeys(self.get_target_subkeys())
        self.attack.setStatsReadyCallback(callback)

        self.attack.addTraces(self.get_trace_source(), self.get_trace_range(), None, pointRange=self.get_point_range())

        # close trace manager
        return self.attack.get_statistics()


    def run(self, callback=None):
        """ Runs the attack

        Args:
            callback (function(), optional): Callback to call every update
                interval. No arguments are passed to callback. Defaults to None.

        Returns:
            Results, the results of the attack. See documentation
            for Results for more details.
        """
        return self.process_traces(callback)
