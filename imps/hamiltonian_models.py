import numpy as np

sigma = [np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
Sx = 0.5*sigma[1]
Sy = 0.5*sigma[2]
Sz = 0.5*sigma[3]
S = [Sx, Sy, Sz]

class XXX:
    def __init__(self, hx = 0, hy = 0, hz = 0):
        self.h = np.array([hx, hy, hz])

    def local_ham(self):
        ham = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
        for i,h in enumerate(self.h):
            if(h != 0):
                ham += np.kron(S[i], sigma[0])
        return ham

    def local_ground_state(self):
        _, eigvectors = np.linalg.eigh(self.local_ham())
        return eigvectors[:,0]
    
    def local_ground_state_en(self):
        eigs, _ = np.linalg.eigh(self.local_ham())
        return eigs[0]

    def hamiltonian(self, L, pbc = False):
        h = self.local_ham()
        ham = np.zeros((int(2**L), int(2**L)), dtype=complex)
        for i in range(L-1):
            ham += np.kron(np.kron(np.identity(int(2**i)), h), np.identity(int(2**(L-i-2))))
        return ham

    def ground_state_en(self, L, pbc = False):
        eigs, _ = np.linalg.eigh(self.hamiltonian(L, pbc))
        return eigs[0]


    def name(self):
        name = "XXX" 
        hstr = ["hx", "hy", "hz"]
        for i,h in enumerate(self.h):
            if(h != 0):
                name += "_" + hstr[i] + str(h)
        return name

class XXZ:
    def __init__(self, delta):
        self.delta = delta

    def local_ham(self):
        ham = np.kron(Sx, Sx) + np.kron(Sy, Sy) + self.delta*np.kron(Sz, Sz)
        return ham

    def hamiltonian(self, L, pbc = False):
        h = self.local_ham()
        ham = np.zeros((int(2**L), int(2**L)), dtype=complex)
        for i in range(L-1):
            ham += np.kron(np.kron(np.identity(int(2**i)), h), np.identity(int(2**(L-i-2))))
        if(pbc):
            bulk = np.identity(int(2**(L-2)))
            ham += np.kron(np.kron(Sx, bulk), Sx)
            ham += np.kron(np.kron(Sy, bulk), Sy)
            ham += self.delta*np.kron(np.kron(Sz, bulk), Sz)
        return ham

    def ground_state_en(self, L, pbc = False):
        eigs, _ = np.linalg.eigh(self.hamiltonian(L, pbc))
        return eigs[0]
