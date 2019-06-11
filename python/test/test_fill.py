import ThrustRTC as trtc



darr = trtc.device_vector('int32_t', 5)
trtc.Fill(darr, trtc.DVInt32(123))
print (darr.to_host())

trtc.Fill(darr, trtc.DVInt32(456), 1,3)
print (darr.to_host())

