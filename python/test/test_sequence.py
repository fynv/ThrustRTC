import ThrustRTC as trtc



darr = trtc.device_vector('int32_t', 10)

trtc.Sequence(darr)
print (darr.to_host())

trtc.Sequence(darr, trtc.DVInt32(1))
print (darr.to_host())

trtc.Sequence(darr, trtc.DVInt32(1), trtc.DVInt32(3))
print (darr.to_host())
