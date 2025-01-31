from chipwhisperer.analyzer.attacks._base import AttackBaseClass
from chipwhisperer.analyzer.attacks.algorithmsbase import AlgorithmsBase
from chipwhisperer.analyzer.attacks.models.base import ModelsBase
import chipwhisperer as cw
cw.open_project

import numpy as np
import time

from .cpa_algorithms.models import get_c_model

class SOCPAAlogrithmBase(AlgorithmsBase):
    """
    Second Order CPA Attack
    """

    def __init__(self):
        super().__init__()
        self.window_size = 300


    def addTraces(self, traceSource, tracerange, progressBar=None, pointRange=None):
        import time
        numtraces = tracerange[1] - tracerange[0]
        numpoints = pointRange[1] - pointRange[0]

        byte_len = max(self.brange) + 1


        full_trace = np.array([traceSource.get_trace(t)[pointRange[0]:pointRange[1]] for t in range(tracerange[0], tracerange[1])])

        average_trace = np.mean(full_trace, axis=0)
        print(average_trace.shape, average_trace.ndim)

        model = get_c_model(self.model)

        from .cpa_algorithms.socpa_kernel import ProductCombineSOCPA
        socpa = ProductCombineSOCPA(byte_len, numpoints, self.window_size, model, average_trace)

        tstart = 0
        tend = self._reportingInterval

        while tstart < numtraces:
            print("tstart: ", tstart)
            if tend > numtraces:
                tend = numtraces

            if tstart > numtraces:
                tstart = numtraces

            trange = range(tstart, tend)
            part_trace = full_trace[tstart:tend]
            part_textin = np.array([traceSource.get_textin(t + tracerange[0]) for t in trange])
            if type(traceSource.get_textout(0)) == bytes:
                part_textout = np.array([np.frombuffer(traceSource.get_textout(t + tracerange[0]), dtype=np.uint8) for t in trange])
            else:
                part_textout = np.array([traceSource.get_textout(t + tracerange[0]) for t in trange])

            part_knownkey = np.array([traceSource.get_known_key(t + tracerange[0]) for t in trange])

            start = time.time()
            corr = socpa.calculate_correlation(part_trace, part_textin, part_textout, part_knownkey)
            end = time.time()
            print("Time taken: ", end-start)

            max_corr_idx = np.argmax(corr, axis=3)
            max_corr = np.max(np.abs(corr), axis=3)
            for bnum in self.brange:
                self.stats.update_subkey(bnum, max_corr[bnum], tnum=tend)
            # del diff
            #     # if progressBar and progressBar.wasAborted():
            #     #     return


            tend += self._reportingInterval
            tstart += self._reportingInterval
            del part_trace, part_textin, part_textout, part_knownkey

            if self.sr:
                self.sr()

        del socpa

        return corr
   

class SOCPA(AttackBaseClass):
    """Second Order CPA Attack"""

    def __init__(self, proj, algorithm : SOCPAAlogrithmBase, leak_model : ModelsBase):
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


    def get_trace_range(self):
        return (self.get_trace_start(), self.get_trace_end())

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
        self.attack.setReportingInterval(self.get_reporting_interval())

        self.attack.setTargetSubkeys(self.get_target_subkeys())
        self.attack.setStatsReadyCallback(callback)

        corr = self.attack.addTraces(self.get_trace_source(), self.get_trace_range(), None, pointRange=self.get_point_range())

        # close trace manager
   
        # return self.attack.get_statistics()
        return corr


