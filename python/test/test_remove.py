import ThrustRTC as trtc

ctx = trtc.Context()

d_value = trtc.device_vector_from_list(ctx, [ 3, 1, 4, 1, 5, 9 ], 'int32_t')
count = trtc.Remove(ctx, d_value, trtc.DVInt32(1))
print (d_value.to_host(0, count))

d_in = trtc.device_vector_from_list(ctx, [ -2, 0, -1, 0, 1, 2 ], 'int32_t')
d_out = trtc.device_vector(ctx, 'int32_t', 6)
count = trtc.Remove_Copy(ctx, d_in, d_out, trtc.DVInt32(0))
print (d_out.to_host(0, count))

is_even = trtc.Functor( ctx, {}, ['x'], 
'''
         return x % 2 == 0;
''')

d_value = trtc.device_vector_from_list(ctx, [ 1, 4, 2, 8, 5, 7 ], 'int32_t')
count = trtc.Remove_If(ctx, d_value, is_even)
print (d_value.to_host(0, count))

d_in = trtc.device_vector_from_list(ctx, [ -2, 0, -1, 0, 1, 2 ], 'int32_t')
d_out = trtc.device_vector(ctx, 'int32_t', 6)
count = trtc.Remove_Copy_If(ctx, d_in, d_out, is_even)
print (d_out.to_host(0, count))

identity = trtc.Functor( ctx, {}, ['x'], 
'''
         return x;
''')

d_value = trtc.device_vector_from_list(ctx, [ 1, 4, 2, 8, 5, 7  ], 'int32_t')
d_stencil = trtc.device_vector_from_list(ctx, [ 0, 1, 1, 1, 0, 0 ], 'int32_t')
count = trtc.Remove_If_Stencil(ctx, d_value, d_stencil, identity)
print (d_value.to_host(0, count))

d_in = trtc.device_vector_from_list(ctx, [ -2, 0, -1, 0, 1, 2 ], 'int32_t')
d_stencil = trtc.device_vector_from_list(ctx, [ 1, 1, 0, 1, 0, 1 ], 'int32_t')
d_out = trtc.device_vector(ctx, 'int32_t', 6)
count = trtc.Remove_Copy_If_Stencil(ctx, d_in, d_stencil, d_out, identity)
print (d_out.to_host(0, count))
