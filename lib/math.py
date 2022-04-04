import math

def min_pow_2(x):
    if x < 0:
        x = -x
    if x == 0:
        return -126 # TODO? The minumum floating point value
    return math.ceil(math.log(x, 2))

