import ThrustRTC as trtc

ctx = trtc.Context()

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
print(trtc.Reduce(ctx, darr))
print(trtc.Reduce(ctx, darr, trtc.DVInt32(1)))
print(trtc.Reduce(ctx, darr, trtc.DVInt32(-1), trtc.Maximum()))
