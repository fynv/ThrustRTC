import ThrustRTC as trtc



d_value = trtc.device_vector_from_list([ 3, 1, 4, 1, 5, 9 ], 'int32_t')
count = trtc.Remove(d_value, trtc.DVInt32(1))
print (d_value.to_host(0, count))

d_in = trtc.device_vector_from_list([ -2, 0, -1, 0, 1, 2 ], 'int32_t')
d_out = trtc.device_vector('int32_t', 6)
count = trtc.Remove_Copy(d_in, d_out, trtc.DVInt32(0))
print (d_out.to_host(0, count))

is_even = trtc.Functor( {}, ['x'], 
'''
         return x % 2 == 0;
''')

d_value = trtc.device_vector_from_list([ 1, 4, 2, 8, 5, 7 ], 'int32_t')
count = trtc.Remove_If(d_value, is_even)
print (d_value.to_host(0, count))

d_in = trtc.device_vector_from_list([ -2, 0, -1, 0, 1, 2 ], 'int32_t')
d_out = trtc.device_vector('int32_t', 6)
count = trtc.Remove_Copy_If(d_in, d_out, is_even)
print (d_out.to_host(0, count))

d_value = trtc.device_vector_from_list([ 1, 4, 2, 8, 5, 7  ], 'int32_t')
d_stencil = trtc.device_vector_from_list([ 0, 1, 1, 1, 0, 0 ], 'int32_t')
count = trtc.Remove_If_Stencil(d_value, d_stencil, trtc.Identity())
print (d_value.to_host(0, count))

d_in = trtc.device_vector_from_list([ -2, 0, -1, 0, 1, 2 ], 'int32_t')
d_stencil = trtc.device_vector_from_list([ 1, 1, 0, 1, 0, 1 ], 'int32_t')
d_out = trtc.device_vector('int32_t', 6)
count = trtc.Remove_Copy_If_Stencil(d_in, d_stencil, d_out, trtc.Identity())
print (d_out.to_host(0, count))
