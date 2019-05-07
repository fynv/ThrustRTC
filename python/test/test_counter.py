import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

negate = trtc.Functor( ctx, {}, ['x'],
'''
         return -x;
''')


darr = trtc.device_vector(ctx, 'int32_t', 10)
trtc.Transform(ctx, trtc.DVCounter(ctx, trtc.DVInt32(5), 10), darr, negate)
print (darr.to_host())
