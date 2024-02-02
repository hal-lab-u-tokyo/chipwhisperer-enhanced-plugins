import numpy as np

from chipwhisperer.analyzer.attacks.algorithmsbase import AlgorithmsBase
from chipwhisperer.logging import *


from .models import get_c_model

class FastCPAProgressive(AlgorithmsBase):
    """
    CPA Attack done as a loop, but using an algorithm which can progressively add traces & give output stats
    """
    _name = "Progressive"

    def __init__(self):
        AlgorithmsBase.__init__(self)

        self.getParams().addChildren([
            {'name':'Iteration Mode', 'key':'itmode', 'type':'list', 'values':{'Depth-First':'df', 'Breadth-First':'bf'}, 'value':'bf', 'action':self.updateScript},
            {'name':'Skip when PGE=0', 'key':'checkpge', 'type':'bool', 'value':False, 'action':self.updateScript},
        ])
        self.updateScript()

    def getCpaKernel(self, byte_len, numpoints, model):
        from .cpa_kernel import FastCPA
        return FastCPA(byte_len, numpoints, model)

    def addTraces(self, traceSource, tracerange, progressBar=None, pointRange=None):
        numtraces = tracerange[1] - tracerange[0]
        numpoints = pointRange[1] - pointRange[0]


        byte_len = max(self.brange) + 1

        tstart = 0
        tend = self._reportingInterval

        model = get_c_model(self.model)

        cpa = self.getCpaKernel(byte_len, numpoints, model)

        while tstart < numtraces:
            if tend > numtraces:
                tend = numtraces

            if tstart > numtraces:
                tstart = numtraces

            trange = range(tstart, tend)
            part_trace = np.array([traceSource.get_trace(t + tracerange[0])[pointRange[0]:pointRange[1]] for t in trange])
            part_textin = np.array([traceSource.get_textin(t + tracerange[0]) for t in trange])
            part_textout = np.array([traceSource.get_textout(t + tracerange[0]) for t in trange])
            part_knownkey = np.array([traceSource.get_known_key(t + tracerange[0]) for t in trange])

            diff = cpa.calculate_correlation(part_trace, part_textin, part_textout, part_knownkey)

            for bnum in self.brange:
                self.stats.update_subkey(bnum, diff[bnum], tnum=tend)
            del diff
                # if progressBar and progressBar.wasAborted():
                #     return

            tend += self._reportingInterval
            tstart += self._reportingInterval
            del part_trace, part_textin, part_textout, part_knownkey

            if self.sr:
                self.sr()

        del cpa

class FastCPAProgressiveCuda(FastCPAProgressive):
    def getCpaKernel(self, byte_len, numpoints, model):
        from .cpa_cuda_kernel import FastCPACuda
        return FastCPACuda(byte_len, numpoints, model)

class FastCPAProgressiveOpenCL(FastCPAProgressive):
    def getCpaKernel(self, byte_len, numpoints, model):
        from .cpa_opencl_kernel import FastCPAOpenCL
        return FastCPAOpenCL(byte_len, numpoints, model)

class FastCPAProgressiveOpenCLFP32(FastCPAProgressive):
    def getCpaKernel(self, byte_len, numpoints, model):
        from .cpa_opencl_kernel import FastCPAOpenCLFP32
        return FastCPAOpenCLFP32(byte_len, numpoints, model)