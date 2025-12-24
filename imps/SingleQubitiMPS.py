import numpy as np
import scipy
from scipy.stats import unitary_group

from . import MPSTransferMatrix as tm
from . import SingularValueStats as svstats

class SingleQubitiMPS:
    # A is assumed to be a (chi, 2, chi) shaped tensor
    def __init__(self, A):
        self.A = A
        self.chi = A.shape[0]

        self.tobj = self.transfer_matrix()
        self.eig()

    def transfer_matrix(self):
        dim = self.chi**2
        T = np.reshape(np.einsum('abd,cbe', self.A, np.conjugate(self.A)), (dim, dim))
        tobj = tm.MPSTransferMatrix(T, self.chi)
        return tobj

    def eig(self):
        self.tobj.eig(True, True)
        leading_eig = self.tobj.eigvals[-1]
        self.A /= np.sqrt(leading_eig)
        self.tobj.T /= leading_eig
        self.tobj.eigvals /= leading_eig
        self.leftvec = self.tobj.leftvecs[-1].reshape((self.chi, self.chi))
        self.rightvec = self.tobj.rightvecs[:,-1].reshape((self.chi, self.chi))

    # gate is a 2x2 matrix
    def apply_gate(self, gate):
        self.A = np.einsum('bc,acd', gate, self.A)
        self.tobj = self.transfer_matrix()
        self.eig()

    def expectation_value(self, operator):
        return np.einsum('ab,acd,ce,bef,df', 
                         self.leftvec, 
                         self.A, 
                         operator, 
                         np.conjugate(self.A), 
                         self.rightvec)

    def density_matrix(self):
        return np.einsum('ab,acd,bef,df', 
                         self.leftvec,
                         self.A, 
                         np.conjugate(self.A),
                         self.rightvec)

    def purity(self):
        rho = self.density_matrix()
        return np.real(np.trace(rho @ rho))

    def magnetization(self):
        Sz = np.array([[1,0],[0,-1]])/2
        return np.real(self.expectation_value(Sz))

    def norm(self):
        return np.real(self.expectation_value(np.eye(2)))

def random_iMPS(chi, orthogonality = 'right'):
    if(orthogonality == 'right'):
        A = np.reshape(unitary_group.rvs(2*chi), (2, chi, 2, chi))[0,:,:,:]
    elif(orthogonality == 'left'):
        A = np.reshape(unitary_group.rvs(2*chi), (chi, 2, chi, 2))[:,:,:,0]
    return SingleQubitiMPS(A)

