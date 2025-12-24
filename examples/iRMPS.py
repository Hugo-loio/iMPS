import numpy as np

from imps import SingleQubitiMPS as iMPS
from imps import SecondMutualInfo as I2

# Compute the second mutual information associated
# to a RMPS tensor (see https://doi.org/10.1103/ymzz-923j)

chi = 10
rs = np.arange(1, 10)

state = iMPS.random_iMPS(chi)
I2obj = I2.SecondMutualInfo(state)

for r in rs:
    print("r =", r, ": I2 =", I2obj.mutual_information(r)[1])
