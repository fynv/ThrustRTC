import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
print(trtc.Reduce(ctx, darr))
