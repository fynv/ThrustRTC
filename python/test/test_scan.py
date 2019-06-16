import ThrustRTC as trtc

darr = trtc.device_vector_from_list([1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Inclusive_Scan(darr, darr)
print (darr.to_host())

darr = trtc.device_vector_from_list([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], 'int32_t')
trtc.Inclusive_Scan(darr, darr, trtc.Maximum())
print (darr.to_host())

darr = trtc.device_vector_from_list([1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Exclusive_Scan(darr, darr)
print (darr.to_host())

darr = trtc.device_vector_from_list([1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Exclusive_Scan(darr, darr, trtc.DVInt32(4))
print (darr.to_host())

darr = trtc.device_vector_from_list([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], 'int32_t')
trtc.Exclusive_Scan(darr, darr, trtc.DVInt32(1), trtc.Maximum())
print (darr.to_host())
