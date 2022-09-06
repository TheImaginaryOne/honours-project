from lib.layer_tracker import HistogramTracker

import torch

# Run with pytest

def test_layer_hist_basic():
    tracker = HistogramTracker(2)
    
    tensor = torch.Tensor([0.5, 1.5, 1.75, -1.2])
    tracker.update(tensor)

    assert tracker.range_pow_2 == 1.
    assert torch.equal(tracker.histogram, torch.Tensor([1, 0, 1, 2]))

def test_layer_hist_resize_simple():
    tracker = HistogramTracker(2)
    
    tensor = torch.Tensor([0.5, 0.25, 0.1, -0.2])
    tracker.update(tensor)
    tensor = torch.Tensor([-3., 5., -6., -7.])
    tracker.update(tensor)

    assert tracker.range_pow_2 == 3.
    assert torch.equal(tracker.histogram, torch.Tensor([2, 2, 3, 1]))

def test_layer_hist_resize_range():
    tracker = HistogramTracker(3)
    
    tensor = torch.Tensor([-4. + i * 0.1 for i in range(80)])
    tracker.update(tensor)
    tensor = torch.Tensor([-8. + i * 0.2 for i in range(80)])
    tracker.update(tensor)

    assert tracker.range_pow_2 == 3.
    assert torch.equal(tracker.histogram, torch.Tensor([10, 10, 30, 30, 30, 30, 10, 10]))


def test_layer_hist_resize_2():
    tracker = HistogramTracker(2)
    
    tensor = torch.Tensor([0.5, 1.5, 1.75, -1.2])
    tracker.update(tensor)
    tensor = torch.Tensor([0.2, -0.2, -1.7, -1.5])
    tracker.update(tensor)

    assert tracker.range_pow_2 == 1.
    assert torch.equal(tracker.histogram, torch.Tensor([3, 1, 2, 2]))
