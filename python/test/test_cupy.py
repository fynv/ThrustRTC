import ThrustRTC as trtc
import numpy as np
import cupy as cp

nparr = np.array([1, 0, 2, 2, 1, 3], dtype=np.int32)
cparr = cp.array(nparr)
darr = trtc.DVCupyVector(cparr)
trtc.Inclusive_Scan(darr, darr)
print(cp.asnumpy(cparr))
