import numpy as np
import cupy as cp
import ThrustRTC as trtc

cparr = cp.empty(4, dtype = np.complex64)
darr = trtc.DVCupyVector(cparr)
c = complex(1.0, 2.0)
trtc.Fill(darr, trtc.DVComplex64(c))
print("input: ", cp.asnumpy(cparr))
cparr = cp.fft.fft(cparr)
# darr = trtc.DVCupyVector(cparr)
print("output: ", cp.asnumpy(cparr))
