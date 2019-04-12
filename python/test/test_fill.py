import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

darr = trtc.device_vector(ctx, 'int32_t', 5)
trtc.Fill(ctx, darr, trtc.DVInt32(123))
print (darr.to_host())

trtc.Fill(ctx, darr, trtc.DVInt32(456), 1,3)
print (darr.to_host())

