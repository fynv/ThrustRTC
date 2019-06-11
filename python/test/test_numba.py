import ThrustRTC as trtc
import numpy as np
from numba import cuda

nparr = np.array([1, 0, 2, 2, 1, 3], dtype=np.int32)
nbarr = cuda.to_device(nparr)
darr = trtc.DVNumbaVector(nbarr)
trtc.Inclusive_Scan(darr, darr)
print(nbarr.copy_to_host())
