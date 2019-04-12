import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

darr = trtc.device_vector(ctx, 'int32_t', 10)

trtc.Sequence(ctx, darr)
print (darr.to_host())

trtc.Sequence(ctx, darr, trtc.DVInt32(1))
print (darr.to_host())

trtc.Sequence(ctx, darr, trtc.DVInt32(1), trtc.DVInt32(3))
print (darr.to_host())
