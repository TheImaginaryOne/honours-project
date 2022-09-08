from typing import List
import torch
from lib.math_utils import min_pow_2
import numpy as np

class Histogram:
    def __init__(self, range_pow_2, values):
        # The histogram covers the numbers from -2^range_pow_2 to 2^range_pow_2
        self.range_pow_2 = range_pow_2
        # array of counts for each bin. We assume it is an even number
        self.values = values

        self.cdf = np.concatenate(([0], np.cumsum(values) / np.sum(values)))
        # x coordinates of cdf
        self.cdf_x = np.linspace(
            -(2. ** self.range_pow_2),
             2. ** self.range_pow_2,
             len(values) + 1)
    
    def percentile(self, p: List[int]):
        """
        Get values for percentile. May be a floating point
        May not give correct values for p = 0 or 1
        """
        return np.interp(p, self.cdf, self.cdf_x)
    
    def minmax(self):
        min_bin = np.nonzero(self.values)[0][0]
        max_bin = np.nonzero(self.values)[0][-1]

        return self.cdf_x[min_bin], self.cdf_x[max_bin + 1]

class HistogramTracker:
    def __init__(self, bin_count_pow_2=8):
        # bin range from -2**range_pow_2 to +2**range_pow_2
        self.range_pow_2 = None
        # have 2 ** bin_count_pow_2 bins.
        self.bin_count_pow_2 = bin_count_pow_2
        self.histogram = None

    def update(self, input):
        """ Update histogram """
        min_val, max_val = torch.aminmax(input)
        # assume symmetric range representation
        max_input_pow_2 = max(min_pow_2(float(min_val)), min_pow_2(float(max_val)))
        if self.range_pow_2 is not None:
            max_input_pow_2 = max(max_input_pow_2, self.range_pow_2)

        bin_count = 2 ** self.bin_count_pow_2
        hist = torch.histc(input, bins=bin_count, 
                min=-2**max_input_pow_2,
                max=2**max_input_pow_2)

        if self.range_pow_2 is None:
            self.histogram = hist
            self.range_pow_2 = max_input_pow_2

        else:
            new_range_pow_2 = max_input_pow_2
            # adjust scale of histogram
            if new_range_pow_2 > self.range_pow_2:

                # scale down the old histogram by 2**scale_factor_pow_2.
                # Note the first 
                scale_factor_pow_2 = min(new_range_pow_2 - self.range_pow_2, 
                        self.bin_count_pow_2 - 1)

                # sum windows of every scale_factor elements
                scaled_hist = torch.sum(torch.reshape(self.histogram, (-1, 2**scale_factor_pow_2)), 1)

                add_hist = torch.zeros(self.histogram.shape)
                add_hist[bin_count//2 - scaled_hist.shape[0]//2
                        :bin_count//2 + scaled_hist.shape[0]//2] = scaled_hist
            else:
                add_hist = self.histogram

            # add histogram
            hist = hist + add_hist
            self.histogram = hist
            self.range_pow_2 = max_input_pow_2

def decide_bounds_percentile(histogram: Histogram, tail: float):
    cdf = np.cumsum(histogram.values) / np.sum(histogram.values)
    #print(cdf)
    return histogram.percentile([tail, 1. - tail])

def decide_bounds_min_max(histogram: Histogram):
    return histogram.minmax()