import ThrustRTC as trtc



darr = trtc.device_vector_from_list([1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Transform_Inclusive_Scan(darr, darr, trtc.Negate(), trtc.Plus())
print (darr.to_host())

darr = trtc.device_vector_from_list([1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Transform_Exclusive_Scan(darr, darr, trtc.Negate(), trtc.DVInt32(4), trtc.Plus())
print (darr.to_host())

