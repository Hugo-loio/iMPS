import numpy as np
import scipy

from . import SingularValueStats as svstats

class MPSTransferMatrix:
    def __init__(self, transfermatrix, chi):
        self.T = transfermatrix
        self.chi = chi

        # The eigenvalues and eigenvectors are sorted in the same way
        # leftvecs contains the left eigenvectors in the lines
        # rightvecs contains the right eigenvectors in the columns

        self.eigvals = None
        self.got_vals = False
        self.leftvecs = None
        self.got_left = False
        self.rightvecs = None
        self.got_right = False

    def eig(self, right = False, left = False, sort_method = 'abs', verbose = False):
        if(right == left == False):
            self.eigvals = np.linalg.eigvals(self.T)
        elif(right == True and left == False):
            self.eigvals, self.rightvecs = scipy.linalg.eig(self.T, right = right, left = left)
            self.got_right = True
        elif(right == False and left == True):
            self.eigvals, self.leftvecs = scipy.linalg.eig(self.T, right = right, left = left)
            self.got_left = True
        elif(right == True and left == True):
            self.eigvals, self.leftvecs, self.rightvecs = scipy.linalg.eig(self.T, right = right, left = left)
            self.got_left = True
            self.got_right = True
        self.got_vals = True

        # Normalize and sort
        match sort_method:
            case 'abs':
                sort = np.argsort(np.abs(self.eigvals))
            case 'real':
                sort = np.argsort(np.real(self.eigvals))
            case 'abs_real':
                sort = np.argsort(np.abs(np.real(self.eigvals)))
            case _:
                raise ValueError('sort_method not valid')

        self.eigvals = self.eigvals[sort]

        if(left == True):
            self.leftvecs = np.conjugate(self.leftvecs[:,sort].T)
        if(right == True):
            self.rightvecs = self.rightvecs[:,sort]
        if(right == left == True):
            norm = np.diag(self.leftvecs @ self.rightvecs)
            self.rightvecs /= norm

        if(verbose):
            #print(np.linalg.norm(self.rightvecs @ np.diag(self.eigvals) @ np.linalg.inv(self.rightvecs) - self.T))
            #print(np.linalg.norm(np.linalg.inv(self.leftvecs) @ np.diag(self.eigvals) @ self.leftvecs - self.T))
            print(np.linalg.norm(self.leftvecs @ self.rightvecs - np.eye(np.square(self.chi))))
            for i in range(np.square(self.chi)):
                for j in range(np.square(self.chi)):
                    val = np.dot(self.leftvecs[i], self.rightvecs[:,j])
                    if(i == j and np.abs(val - 1) > 1E-2):
                        print(i,j, val)
                    elif(i != j and np.abs(val) > 1E-2):
                        print(i,j, val)
                        print(self.eigvals[i], self.eigvals[j])

    def check_left_right_normalization(self):
        return np.allclose(np.eye(self.eigvals.shape[0]), self.leftvecs @ self.rightvecs)

    def get_eigvals(self):
        if(self.got_vals == False):
            self.eig()
        return self.eigvals

    # These gap functions are not very good, but the eigenvalues should be properly sorted
    def gap(self):
        eigvals = np.sort(self.get_eigvals().real)
        return eigvals[-1] - eigvals[-2]

    def gap2(self):
        mags = np.sort(np.abs(self.get_eigvals()))
        return mags[-1] - mags[-2]

    def check_orthogonality(self):
        identity = np.eye(self.chi)
        T = self.T.reshape((self.chi, self.chi, self.chi, self.chi))
        left_orthogonal = np.allclose(identity, np.einsum('ab,abcd', identity, T))
        right_orthogonal = np.allclose(identity, np.einsum('abcd,cd', T, identity))
        return left_orthogonal, right_orthogonal

    def vecs_svdvals(self, remove_null_space = False):
        if(self.got_left == self.got_right == False):
            self.eig(right = True)
        vecs = self.rightvecs
        if(self.got_right == False):
            vecs = self.leftvecs

        if(remove_null_space):
            vals_abs = np.abs(self.eigvals)
            sort = np.argsort(vals_abs)
            vals_abs = vals_abs[sort]
            index = np.argmax(vals_abs >  1E-11)
            vecs = vecs[:,sort[index:]]

        return svstats.SingularValueStats(np.linalg.svd(vecs, compute_uv = False))
