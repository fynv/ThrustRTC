import ThrustRTC as trtc



darr1_in = trtc.device_vector_from_list([ 1, 2, 1, 2, 1, 2, 1, 2], 'int32_t')
darr1_out = trtc.device_vector('int32_t', 8)
trtc.Adjacent_Difference(darr1_in, darr1_out)
print (darr1_out.to_host())

darr2_in = trtc.device_vector_from_list([ 1, 2, 1, 2, 1, 2, 1, 2], 'int32_t')
darr2_out = trtc.device_vector('int32_t', 8)

trtc.Adjacent_Difference(darr2_in, darr2_out, trtc.Plus())
print (darr2_out.to_host())

