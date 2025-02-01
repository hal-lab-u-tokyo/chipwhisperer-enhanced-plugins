###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /lib/cw_plugins/analyzer/attacks/socpa_stats.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  01-02-2025 08:21:46
#   Last Modified: 01-02-2025 09:02:32
###


import numpy as np

class SOCPAResults(object):
    def __init__(self, num_subkeys, num_perms, window_size):
        self.window_size = window_size
        self.num_subkeys = num_subkeys
        self.num_perms = num_perms
        self.known_key = None
        self.clear()


    def clear(self):
        self.corr = [None] * self.num_subkeys

        # for calculated result cache
        ### maxes[byte_index][rank] contatins [hypothesis key, p0, p1, correlation]
        self.maxes = [np.zeros(self.num_perms, dtype=[('hyp', 'i2'), ('point', 'i4'), ('corr', 'f8')]) for _ in range(self.num_subkeys) ]
        self.max_pos = [np.zeros((self.num_perms, 2), dtype=np.uint)] * self.num_subkeys
        self.max_ready = [False]*self.num_subkeys

        self.pge = [self.num_perms - 1] * self.num_subkeys



    def __str__(self):
        ret = ""
        ret += "\tSubkey\tKGuess\tSamplePos\tCorrelation\tPGE\n"
        guesses = self.best_guesses()
        for i,result in enumerate(guesses):
            # p0, p1 = result["pos"]
            hyp = result["guess"]
            pos = self.max_pos[i][hyp]
            ret += f"\t{i:02d}\t0x{hyp:02X}\t({pos[0]:4d},{pos[1]:4d})\t{result['correlation']:7.5f}\t\t{result['pge']}\n"
        return ret


    def update_subkey(self, bnum, data, tnum):
        """
        Update the subkey with new data
        """
        self.corr[bnum] = data[:]
        # new data is added, so we need to recalculate the maximums
        self.max_ready[bnum] = False

    def find_key(self, use_absolute=True):
        maxes = self.find_maximums(use_absolute)
        return [max_subkey[0]["hyp"] for max_subkey in maxes]

    def best_guesses(self):
        guess_list = []
        maxes = self.find_maximums()
        for i, subkey_result in enumerate(maxes):
            guess = {}
            guess['guess'] = subkey_result[0]["hyp"]
            guess['correlation'] = subkey_result[0]["corr"]
            guess['pge'] = self.pge[i]
            guess["pos"] = self.max_pos[i][guess['guess']]
            guess_list.append(guess)

        return guess_list


    def find_maximums(self, use_absolute=True, use_single=False):

        for byte_index in range(self.num_subkeys):
            if self.max_ready[byte_index]:
                continue

            for hyp in range(self.num_perms):
                # 2D array [0:num_points][0:num_window]
                corr = self.corr[byte_index][hyp]

                if use_absolute:
                    # reduction along the window axis
                    max_corr_window = np.nanmax(np.fabs(corr), axis=1)
                    max_win_pos = np.nanargmax(np.fabs(corr), axis=1)
                else:
                    # reduction along the window axis
                    max_corr_window = np.nanmax(corr, axis=1)
                    max_win_pos = np.nanargmax(corr, axis=1)


                # reduction along the trace sample axis
                max_corr = np.nanmax(max_corr_window)
                max_sample_pos = np.nanargmax(max_corr_window)

                # max combine point
                max_comb_point = max_win_pos[max_sample_pos] + max_sample_pos + 1

                self.maxes[byte_index][hyp]["hyp"] = hyp
                self.maxes[byte_index][hyp]["corr"] = max_corr
                self.maxes[byte_index][hyp]["point"] = max_sample_pos
                self.max_pos[byte_index][hyp] = (max_sample_pos, max_comb_point)

            self.maxes[byte_index][::-1].sort(order='corr')
            self.max_ready[byte_index] = True

            if self.known_key is not None:
                self.pge[byte_index] = np.where(self.maxes[byte_index]["hyp"] == self.known_key[byte_index])[0][0]


        return self.maxes

    def set_known_key(self, known_key):
        self.known_key = known_key