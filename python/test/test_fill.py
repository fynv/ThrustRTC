import ThrustRTC as trtc


darr = trtc.device_vector('int32_t', 5)
trtc.Fill(darr, trtc.DVInt32(123))
print (darr.to_host())

trtc.Fill(darr.range(1,3), trtc.DVInt32(456))
print (darr.to_host())

