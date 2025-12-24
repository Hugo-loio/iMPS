import numpy as np

class FramePotentialCalculator:
    def __init__(self, chi):
        self.chi = chi
        #self.A2 = np.zeros(((2*chi)**2, chi**2), dtype=complex)
        self.A2 = np.zeros((chi**2, (2*chi)**2), dtype=complex)
        self.num_samples = 0

    def add(self, A):
        #M = np.reshape(A, (2*self.chi, self.chi))
        M = np.reshape(A, (self.chi, 2*self.chi))
        self.A2 += np.kron(np.conjugate(M),M)
        self.num_samples += 1

    def frame_potential(self):
        return np.square(np.linalg.norm(self.A2)/self.num_samples)

def safe_log(x):
    if (np.abs(x) < 1E-9):
        x = 1E-9
    return np.log(x)
