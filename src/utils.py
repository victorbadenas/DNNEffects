import time

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
