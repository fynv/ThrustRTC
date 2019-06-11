import ThrustRTC as trtc



darr = trtc.device_vector('int32_t', 10)

trtc.Sequence(darr)
trtc.Tabulate(darr, trtc.Negate())
print (darr.to_host())
