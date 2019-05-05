import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

maximum = trtc.Functor( {}, ['x','y'], 'ret',
'''
         ret = x>y?x:y;
''')

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Inclusive_Scan(ctx, darr, darr)
print (darr.to_host())

darr = trtc.device_vector_from_list(ctx, [-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], 'int32_t')
trtc.Inclusive_Scan(ctx, darr, darr, maximum)
print (darr.to_host())