import ThrustRTC as trtc
dIn = trtc.device_vector_from_list([ 10, 20, 30, 40, 50, 60, 70, 80 ], 'int32_t')
dRange = dIn.range(2, 6)
dOut = trtc.device_vector('int32_t', 4)

trtc.Copy(dRange, dOut)
print (dOut.to_host())
