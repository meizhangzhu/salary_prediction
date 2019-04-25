import json
import code
import collections

import numpy as np
import sklearn as sk
import sklearn.metrics

def metric_is_improving(metric_history, history_len=2):
    if len(metric_history) < history_len:
        return True

    improving = False
    recent_history = metric_history[-history_len:]
    for idx in range(1,len(recent_history)):
        if recent_history[idx] < recent_history[idx-1]:
            improving = True
    return improving

class StatisticsReporter():
    def __init__(self):
        self.statistics = collections.defaultdict(list)

    def update_data(self, d):
        for k, v in d.items():
            self.statistics[k] += [v]

    def clear(self):
        self.statistics = collections.defaultdict(list)

    def to_string(self):
        string_list = []
        for k, v in self.statistics.items():
            mean = np.mean(v)
            string_list.append("{}: {:.5g}".format(k, mean))
        return ", ".join(string_list)

    def get_value(self, key):
        if key in self.statistics:
            value = np.mean(self.statistics[key])
            return value
        else:
            return None