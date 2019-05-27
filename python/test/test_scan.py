import ThrustRTC as trtc

ctx = trtc.Context()

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Inclusive_Scan(ctx, darr, darr)
print (darr.to_host())

darr = trtc.device_vector_from_list(ctx, [-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], 'int32_t')
trtc.Inclusive_Scan(ctx, darr, darr, trtc.Maximum())
print (darr.to_host())

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Exclusive_Scan(ctx, darr, darr)
print (darr.to_host())

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Exclusive_Scan(ctx, darr, darr, trtc.DVInt32(4))
print (darr.to_host())

darr = trtc.device_vector_from_list(ctx, [-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], 'int32_t')
trtc.Exclusive_Scan(ctx, darr, darr, trtc.DVInt32(1), trtc.Maximum())
print (darr.to_host())
