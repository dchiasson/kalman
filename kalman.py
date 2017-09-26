import numpy as np

class Kalman(object):
    def __init__(self, A, H, B, *args, **kwargs):
        self.A = np.matrix(A)
        self.H = np.matrix(H)
        self.B = np.matrix(B)
        kwargs.has_key()
