import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

darr1 = trtc.device_vector_from_list(ctx, [1,2,3,1,2], 'int32_t')
trtc.Replace(ctx, darr1, trtc.DVInt32(1), trtc.DVInt32(99))
print (darr1.to_host())

darr2 = trtc.device_vector_from_list(ctx, [1, -2, 3, -4, 5 ], 'int32_t')
trtc.Replace_If(ctx, darr2, trtc.Functor( {}, ['x'], 'ret',
'''
         ret = x<0;
'''), trtc.DVInt32(0))
print (darr2.to_host())

darr3_in = trtc.device_vector_from_list(ctx, [1,2,3,1,2], 'int32_t')
darr3_out = trtc.device_vector(ctx, 'int32_t', 5)
trtc.Replace_Copy(ctx, darr3_in, darr3_out, trtc.DVInt32(1), trtc.DVInt32(99))
print (darr3_out.to_host())

darr4_in = trtc.device_vector_from_list(ctx, [1, -2, 3, -4, 5 ], 'int32_t')
darr4_out = trtc.device_vector(ctx, 'int32_t', 5)
trtc.Replace_Copy_If(ctx, darr4_in, darr4_out, trtc.Functor( {}, ['x'], 'ret',
'''
         ret = x<0;
'''), trtc.DVInt32(0))
print (darr4_out.to_host())
