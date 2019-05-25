import ThrustRTC as trtc

ctx = trtc.Context()

is_even = trtc.Functor( ctx, {}, ['x'], 
'''
         return x % 2 == 0;
''')

d_value = trtc.device_vector_from_list(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
trtc.Partition(ctx, d_value, is_even)
print (d_value.to_host())

d_value = trtc.device_vector_from_list(ctx, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'int32_t')
d_stencil = trtc.device_vector_from_list(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
trtc.Partition_Stencil(ctx, d_value, d_stencil, is_even)
print (d_value.to_host())

d_value = trtc.device_vector_from_list(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
d_evens = trtc.device_vector(ctx, 'int32_t', 10)
d_odds = trtc.device_vector(ctx, 'int32_t', 10)
count = trtc.Partition_Copy(ctx, d_value, d_evens, d_odds, is_even)
print (d_evens.to_host(0, count))
print (d_odds.to_host(0, 10-count))

d_value = trtc.device_vector_from_list(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
d_stencil = trtc.device_vector_from_list(ctx, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'int32_t')
d_evens = trtc.device_vector(ctx, 'int32_t', 10)
d_odds = trtc.device_vector(ctx, 'int32_t', 10)
count = trtc.Partition_Copy_Stencil(ctx, d_value, d_stencil, d_evens, d_odds, trtc.Identity())
print (d_evens.to_host(0, count))
print (d_odds.to_host(0, 10-count))
