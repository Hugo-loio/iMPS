import numpy as np
from scipy.stats import unitary_group

from . import MPSTransferMatrix as tm

class TwoQubitiMPS:
    # init_state should be a two site state 
    def __init__(self, init_state, bond_dimension, error = 1e-11):
        self.chi = bond_dimension
        self.error = error
        self.sigmaB = np.array([1])
        self.sigmaB_inv = self.sigmaB.copy()
        self.chiB = 1

        self.A, self.sigmaA, self.B = np.linalg.svd(np.reshape(init_state, (2,2)))
        self.sigmaA = self.sigmaA[:self.chi]
        self.sigmaA = np.array([s for s in self.sigmaA if s > self.error])
        self.sigmaA = self.sigmaA/np.linalg.norm(self.sigmaA)
        self.chiA = len(self.sigmaA)
        self.sigmaA_inv = np.reciprocal(self.sigmaA)
        self.A = self.A[:,0:self.chiA]
        self.B = self.B[0:self.chiA,:]
        self.A = self.A[np.newaxis,:,:]
        self.B = self.B[:,:,np.newaxis]
        # Saves the direction of the canonical form
        self.canonical_form_dir = 'AB'
        # This list is ordered based on the canonical form
        self.MPS_objs = [self.A, self.sigmaA, self.B, self.sigmaB]

    def TEBD_AB_gate(self, gate, truncate = True):
        auxA = self.sigmaB[:, np.newaxis, np.newaxis] * self.A * self.sigmaA
        auxB = self.B * self.sigmaB
        M = np.einsum('bcef,aeg,gfd', gate, auxA, auxB).reshape((2*self.chiB,2*self.chiB))
        self.A, self.sigmaA, self.B = np.linalg.svd(M, full_matrices = False)
        self.sigmaA = np.array([s for s in self.sigmaA if s > self.error])
        if(truncate):
            self.sigmaA = self.sigmaA[:self.chi]
        self.sigmaA = self.sigmaA/np.linalg.norm(self.sigmaA)
        self.chiA = len(self.sigmaA)
        self.A = self.A[:,0:self.chiA]
        self.B = self.B[0:self.chiA,:]
        self.sigmaA_inv = np.reciprocal(self.sigmaA)
        self.A = self.sigmaB_inv[:,np.newaxis,np.newaxis] * self.A.reshape(self.chiB,2,self.chiA)
        self.B = self.B.reshape(self.chiA, 2, self.chiB) * self.sigmaB_inv
        self.canonical_form_dir = 'AB'
        self.MPS_objs = [self.A, self.sigmaA, self.B, self.sigmaB]

    def TEBD_BA_gate(self, gate, truncate = True):
        auxB = self.sigmaA[:, np.newaxis, np.newaxis] * self.B * self.sigmaB
        auxA = self.A * self.sigmaA
        M = np.einsum('bcef,aeg,gfd', gate, auxB, auxA).reshape((2*self.chiA,2*self.chiA))
        self.B, self.sigmaB, self.A = np.linalg.svd(M, full_matrices = False)
        self.sigmaB = np.array([s for s in self.sigmaB if s > self.error])
        if(truncate):
            self.sigmaB = self.sigmaB[:self.chi]
        self.sigmaB = self.sigmaB/np.linalg.norm(self.sigmaB)
        self.chiB = len(self.sigmaB)
        self.B = self.B[:,0:self.chiB]
        self.A = self.A[0:self.chiB,:]
        self.sigmaB_inv = np.reciprocal(self.sigmaB)
        self.B = self.sigmaA_inv[:, np.newaxis, np.newaxis] * self.B.reshape(self.chiA,2,self.chiB)
        self.A = self.A.reshape(self.chiB, 2, self.chiA) * self.sigmaA_inv
        self.canonical_form_dir = 'BA'
        self.MPS_objs = [self.A, self.sigmaA, self.B, self.sigmaB]

    def set_canonical_form(self, direction = 'AB'):
        #print("I was called")
        id_gate = np.kron(np.eye(2), np.eye(2)).reshape((2,2,2,2))
        if(direction == 'AB'):
            self.TEBD_AB_gate(id_gate)
        elif(direction == 'BA'):
            self.TEBD_BA_gate(id_gate)
        else:
            raise ValueError('direction should be either \'AB\' or \'BA\'')

    def local_MPS_tensor(self, site = 'A'):
        chi = np.max([self.chiA, self.chiB])
        res = np.zeros((chi, 2, chi), dtype = self.A.dtype)
        if(site == 'A'):
            if(self.canonical_form_dir == 'AB'):
                res[:self.chiB,:,:self.chiA] = self.sigmaB[:, np.newaxis, np.newaxis] * self.A 
            elif(self.canonical_form_dir == 'BA'):
                res[:self.chiB,:,:self.chiA] = self.A * self.sigmaA
        elif(site == 'B'):
            if(self.canonical_form_dir == 'AB'):
                res[:self.chiA,:,:self.chiB] = self.B * self.sigmaB
            elif(self.canonical_form_dir == 'BA'):
                res[:self.chiA,:,:self.chiB] = self.sigmaA[:, np.newaxis, np.newaxis] * self.B 
        else:
            raise ValueError('Value of site can only correspond to \'A\' or \'B\'')
        return res

    def coarse_grain_MPS_tensor(self):
        tensor1 = self.MPS_objs[0]
        inner_vals = self.MPS_objs[1]
        tensor2 = self.MPS_objs[2]
        outer_vals = self.MPS_objs[3]
        chi = outer_vals.shape[0]
        aux = np.einsum('abc,c,cde', tensor1, inner_vals, tensor2)
        return np.reshape(outer_vals, (chi,1,1,1)) * aux * outer_vals 

    def local_TM(self, site = 'A'):
        aux = self.local_MPS_tensor(site)
        dim = np.square(aux.shape[0])
        T = np.einsum('abd,cbe', np.conjugate(aux), aux).reshape((dim,dim))
        return tm.MPSTransferMatrix(T, aux.shape[0])

    def local_operator_TM(self, op, site = 'A'):
        aux = self.local_MPS_tensor(site)
        dim = np.square(aux.shape[0])
        T = np.einsum('abe,bc,dcf', np.conjugate(aux), op, aux).reshape((dim,dim))
        return tm.MPSTransferMatrix(T, aux.shape[0])

    def check_canonical_form(self):
        original_canonical_form_dir = self.canonical_form_dir

        self.canonical_form_dir = 'AB'
        TA = self.local_TM('A')
        TB = self.local_TM('B')
        print("AB canonical form")
        print("A:", TA.check_unitary()[0], "B:", TB.check_unitary()[1])

        self.canonical_form_dir = 'BA'
        TA = self.local_TM('A')
        TB = self.local_TM('B')
        print("BA canonical form")
        print("A:", TA.check_unitary()[1], "B:", TB.check_unitary()[0])

        self.canonical_form_dir = original_canonical_form_dir

    def expectation_value_AB(self, operator):
        if(self.canonical_form_dir == 'BA'):
            self.set_canonical_form('AB')
        auxA = self.sigmaB[:, np.newaxis, np.newaxis] * self.A * self.sigmaA
        auxB = self.B * self.sigmaB
        return np.einsum('abc,cde,bdfg,afh,hge', np.conjugate(auxA), np.conjugate(auxB), operator, auxA, auxB)

    def expectation_value_BA(self, operator):
        if(self.canonical_form_dir == 'AB'):
            self.set_canonical_form('BA')
        auxB = self.sigmaA[:, np.newaxis, np.newaxis] * self.B * self.sigmaB
        auxA = self.A * self.sigmaA
        return np.einsum('abc,cde,bdfg,afh,hge', np.conjugate(auxB), np.conjugate(auxA), operator, auxB, auxA)

    def density(self, operator):
        if(self.canonical_form_dir == 'AB'):
            evAB = self.expectation_value_AB(operator)
            evBA = self.expectation_value_BA(operator)
        else:
            evBA = self.expectation_value_BA(operator)
            evAB = self.expectation_value_AB(operator)
        return (evAB + evBA)/2

    def norm(self):
        operator = np.eye(4).reshape((2,2,2,2))
        if(self.canonical_form_dir == 'AB'):
            return np.real(self.expectation_value_AB(operator))
        else:
            return np.real(self.expectation_value_BA(operator))

    def reduced_density_matrix(self, site = 'A'):
        tensor = self.coarse_grain_MPS_tensor()
        if(site == self.canonical_form_dir[0]):
            return np.einsum('abcd,aecd', tensor, np.conjugate(tensor))
        elif(site == self.canonical_form_dir[1]):
            return np.einsum('abcd,abed', tensor, np.conjugate(tensor))
        else:
            raise ValueError('Value of site can only correspond to \'A\' or \'B\'')

    def reduced_density_matrix_coarse_grained(self):
        tensor = self.coarse_grain_MPS_tensor()
        return np.einsum('abcd,aefd', tensor, np.conjugate(tensor)).reshape(4,4)

    def purity(self, site = None):
        rho = 0
        if(site == None):
            rho = self.reduced_density_matrix_coarse_grained()
        else:
            rho = self.reduced_density_matrix(site)
        return np.real(np.trace(rho @ rho))
    
    def entanglement_entropies(self):
        sigmaA2 = np.square(self.sigmaA)
        SAB = -np.sum(sigmaA2*np.log(sigmaA2 + 1E-9))
        sigmaB2 = np.square(self.sigmaB)
        SBA = -np.sum(sigmaB2*np.log(sigmaB2 + 1E-9))
        return SAB, SBA

    '''

    # The R TM is right unitary and the L TM is left unitary
    def coarse_grain_TM(self, direction = 'L'):
        vals = self.MPS_objs[3]
        chi = vals.shape[0]
        dim = np.square(chi)
        gamma = self.coarse_grain_MPS_tensor()
        if(direction == 'L'):
            gamma = gamma * vals
        elif(direction == 'R'):
            gamma = vals[:, np.newaxis, np.newaxis] * gamma
        else:
            raise ValueError('Value of direction can only correspond to \'R\' or \'L\'')
        T = np.einsum('abd,cbe', gamma, np.conjugate(gamma)).reshape((dim,dim)) 
        return tm.MPSTransferMatrix(T, chi)

    def normAB(self):
        auxA = self.sigmaB[:, np.newaxis, np.newaxis] * self.A * self.sigmaA
        auxB = self.B * self.sigmaB
        M = np.einsum('abc,cde', auxA, auxB).reshape(np.square(2*self.chiB))
        return np.linalg.norm(M)

    def normBA(self):
        auxB = self.sigmaA[:, np.newaxis, np.newaxis] * self.B * self.sigmaB
        auxA = self.A * self.sigmaA
        M = np.einsum('abc,cde', auxB, auxA).reshape(np.square(2*self.chiA))
        return np.linalg.norm(M)

    # Just switched sites A and B here but should result in the same thing
    def reduced_density_matrix2(self, site = 'A'):
        auxB = self.sigmaA[:, np.newaxis, np.newaxis] * self.B * self.sigmaB
        auxA = self.A * self.sigmaA
        if(site == 'A'):
            return np.einsum('abc,cde,abg,gfe', np.conjugate(auxB), np.conjugate(auxA), auxB, auxA)
        elif(site == 'B'):
            return np.einsum('abc,cde,afg,gde', np.conjugate(auxB), np.conjugate(auxA), auxB, auxA)
        else:
            raise ValueError('Value of site can only correspond to \'A\' or \'B\'')


        '''

