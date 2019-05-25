import ThrustRTC as trtc

ctx = trtc.Context()
darr = trtc.device_vector_from_list(ctx, [3, 7, 2, 5 ], 'int32_t')
trtc.Transform_Binary(ctx, darr, trtc.DVConstant(ctx, trtc.DVInt32(10)), darr, trtc.Plus())
print (darr.to_host())
