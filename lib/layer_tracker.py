import torch
from lib.math_utils import min_pow_2

class Histogram:
    def __init__(self, range_pow_2, values):
        # The histogram covers the numbers from -2^range_pow_2 to 2^range_pow_2
        self.range_pow_2 = range_pow_2
        # array of counts for each bin
        self.values = values

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
