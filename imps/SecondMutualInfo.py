import warnings
warnings.filterwarnings("error")

import numpy as np

from . import SingleQubitiMPS as imps
from .utils import safe_log

class SecondMutualInfo:
    def __init__(self, state : imps.SingleQubitiMPS, compute_exact = True):
        self.state = state

        l1 = state.leftvec
        r1 = state.rightvec
        chi = state.chi

        # Normalize the identity eigenvector
        norm = chi/np.trace(r1)
        r1 = norm*r1
        l1 = l1/norm

        if(not np.allclose(r1, np.eye(chi))):
            raise ValueError("State needs to be right orthogonal!")

        denominator = np.square(np.einsum("ij,ji", l1, l1))

        lambda2 = state.tobj.eigvals[-2]

        l2 = state.tobj.leftvecs[-2].reshape((chi,chi))
        r2 = state.tobj.rightvecs[:,-2].reshape((chi,chi))

        self.dominant_eigval = np.abs(lambda2)

        dominant_overlap = np.einsum("ij,ji", l1, l2)*np.einsum("ab,bc,ac", l1, l1, r2)
        self.dominant_overlaps = np.abs(dominant_overlap/denominator)

        self.lambda2 = lambda2
        self.overlap2 = dominant_overlap/denominator

        # Exact mutual information with all the eigenvectors
        self.compute_exact = compute_exact
        self.Lambdas = []
        self.overlaps = []

        if(compute_exact):
            dim = np.square(chi)
            for i in range(1,dim+1):
                for j in range(1,dim+1):
                    if(i == j == 1):
                        continue
                    Lambda = state.tobj.eigvals[-i]*state.tobj.eigvals[-j]
                    li = state.tobj.leftvecs[-i].reshape((chi,chi))
                    lj = state.tobj.leftvecs[-j].reshape((chi,chi))
                    ri = state.tobj.rightvecs[:,-i].reshape((chi,chi))
                    rj = state.tobj.rightvecs[:,-j].reshape((chi,chi))
                    if(i == 1):
                        li = l1
                        ri = r1
                    if(j == 1):
                        lj = l1
                        rj = r1
                    numerator = np.einsum("ij,ji", lj, li)*np.einsum("ab,ac,db,dc", ri, l1, l1, rj)
                    self.overlaps.append(numerator/denominator)
                    self.Lambdas.append(Lambda)

        """
        if(compute_exact):
            dim = np.square(chi)
            indices = [(i,j) for i in range(1, dim + 1) for j in range(i, dim + 1)]
            for i,j in indices:
                if(i == j == 1):
                    continue
                Lambda = state.tobj.eigvals[-i]*state.tobj.eigvals[-j]
                li = state.tobj.leftvecs[-i].reshape((chi,chi))
                lj = state.tobj.leftvecs[-j].reshape((chi,chi))
                ri = state.tobj.rightvecs[:,-i].reshape((chi,chi))
                rj = state.tobj.rightvecs[:,-j].reshape((chi,chi))
                if(i == 1):
                    li = l1
                    ri = r1
                if(j == 1):
                    lj = l1
                    rj = r1
                numerator = np.einsum("ij,ji", lj, li)*np.einsum("ab,ac,db,dc", ri, l1, l1, rj)
                if(i == j):
                    self.overlaps.append(numerator/denominator)
                else:
                    self.overlaps.append(2*numerator/denominator)
                self.Lambdas.append(Lambda)
                    """

        self.overlaps = np.array(self.overlaps)
        self.Lambdas = np.array(self.Lambdas)

    def mutual_information(self, r):
        try:
            leading_term = 2 * np.real((self.lambda2**r)*self.overlap2)
            if(np.abs(self.lambda2.imag) > 1E-11):
                leading_term *= 2 
            Iapprox = safe_log(1 + leading_term)
        except RuntimeWarning:
            print("Iapprox error")
            print(r, self.dominant_eigval, self.dominant_overlaps)
            Iapprox = np.nan

        if(not self.compute_exact):
            return Iapprox

        try:
            Iexact = safe_log(1 + np.sum((self.Lambdas**r) * self.overlaps).real)
        except RuntimeWarning:
            print("Iexact error")
            print(r, np.sum(self.Lambdas**r * self.overlaps))
            Iexact = np.nan

        return Iapprox, Iexact
