import numpy as np

class SingularValueStats:
    def __init__(self, svdvals):
        self.svdvals = svdvals
        self.svdvals2 = None
        self.normalize()

    def normalize(self):
        self.svdvals /= np.sum(self.svdvals)

    def Renyi_entropy(self, n):
        if(n == 1):
            return -np.sum(self.svdvals * np.log(self.svdvals + 1E-11)) 
        else:
            return (1/(1-n))*np.log(np.sum(np.pow(self.svdvals, n)))

    def IPR(self):
        if(self.svdvals2 == None):
            self.svdvals2 = np.square(self.svdvals)
        return np.sum(self.svdvals2)
