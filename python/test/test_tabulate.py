import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

negate = trtc.Functor( ctx, {}, ['x'], 
'''
         return -x;
''')


darr = trtc.device_vector(ctx, 'int32_t', 10)

trtc.Sequence(ctx, darr)
trtc.Tabulate(ctx, darr, negate)
print (darr.to_host())
