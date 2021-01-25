import time
import math


def timer(print_=False):
    def inner2(func):
        def inner(*args, **kwargs):
            st = time.time()
            ret = func(*args, **kwargs)
            if print_:
                print(f"{func.__name__} ran in {time.time()-st:.2f}s")
                return ret
            else:
                delta = time.time() - st
                return ret, delta
        return inner
    return inner2


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        if self.start_time is not None:
            raise ValueError("timer has not been stopped")
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("timer has not been started")
        time_delta = time.time() - self.start_time
        self.start_time = None
        return time_delta


def UnPool1dOut(Win, s, p, k):
    return (Win-1)*s - 2*p + k


def Lout(Lin, k, d, s, p):
    Lout = Lin + 2*p - d*(k-1) - 1
    Lout /= s
    Lout = int(Lout + 1)
    return Lout


def unpoolParameters(in_size, out_size):
    stride = math.ceil(out_size / (in_size-1))
    tmp = out_size - (in_size - 1) * stride  # tmp= -2*padding + kernel_size
    padding = int(-1 * tmp / 2)
    kernel_size = tmp + 2 * padding
    while padding > 2 * kernel_size or padding < 0:
        padding += 1
        kernel_size = tmp + 2 * padding
    return {'stride': stride, 'padding': padding, 'kernel_size': kernel_size}
