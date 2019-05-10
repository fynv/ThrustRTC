import ThrustRTC as trtc

ctx = trtc.Context()

printf_functor =  trtc.Functor( ctx, {}, ['x'],
'''
         printf("%d\\n", x);
''')

darr = trtc.device_vector_from_list(ctx, [1,2,3,1,2], 'int32_t')
trtc.For_Each(ctx, darr, printf_functor)

