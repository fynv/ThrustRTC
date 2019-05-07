import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

maximum = trtc.Functor( ctx, {}, ['x','y'],
'''
         return x>y?x:y;
''')

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
print(trtc.Reduce(ctx, darr))
print(trtc.Reduce(ctx, darr, trtc.DVInt32(1)))
print(trtc.Reduce(ctx, darr, trtc.DVInt32(-1), maximum))
