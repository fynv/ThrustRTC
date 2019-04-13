import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

printf_functor =  trtc.Functor( {}, ['x'], None,
'''
         printf("%d\\n", x);
''')

darr = trtc.device_vector_from_list(ctx, [1,2,3,1,2], 'int32_t')
trtc.For_Each(ctx, darr, printf_functor)

