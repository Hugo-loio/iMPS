import numpy as np
from scipy.stats import unitary_group

from imps.TwoQubitiMPS import TwoQubitiMPS

# iTEBD evolution of a random Haar TI circuit
# (replicate the simulations of Fig. 2 in 
#  https://doi.org/10.1103/ymzz-923j)

def randU():
    return unitary_group.rvs(4).reshape((2,2,2,2))

T = 20
chi = 10

init_state = np.kron([0,1], [1,0]) # |...101010...> initial state

ts = np.arange(1,T+0.5)

mps = TwoQubitiMPS(init_state, chi)
for t in ts:
    if(t % 2 == 1):
        mps.TEBD_AB_gate(randU())
    else:
        mps.TEBD_BA_gate(randU())

    print("t =", t)
    # You can do something with the spectrum of the local
    # matrix product state tensors...
    TA = mps.local_TM('A')
    eigsA = TA.get_eigvals()
    #TB = mps.local_TM('B')
    #eigsB = TB.get_eigvals()
