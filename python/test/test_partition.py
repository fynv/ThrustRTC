import ThrustRTC as trtc



is_even = trtc.Functor( {}, ['x'], 
'''
         return x % 2 == 0;
''')

d_value = trtc.device_vector_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
trtc.Partition(d_value, is_even)
print (d_value.to_host())

d_value = trtc.device_vector_from_list([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'int32_t')
d_stencil = trtc.device_vector_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
trtc.Partition_Stencil(d_value, d_stencil, is_even)
print (d_value.to_host())

d_value = trtc.device_vector_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
d_evens = trtc.device_vector('int32_t', 10)
d_odds = trtc.device_vector('int32_t', 10)
count = trtc.Partition_Copy(d_value, d_evens, d_odds, is_even)
print (d_evens.to_host(0, count))
print (d_odds.to_host(0, 10-count))

d_value = trtc.device_vector_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32_t')
d_stencil = trtc.device_vector_from_list([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'int32_t')
d_evens = trtc.device_vector('int32_t', 10)
d_odds = trtc.device_vector('int32_t', 10)
count = trtc.Partition_Copy_Stencil(d_value, d_stencil, d_evens, d_odds, trtc.Identity())
print (d_evens.to_host(0, count))
print (d_odds.to_host(0, 10-count))

d_value = trtc.device_vector_from_list([ 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 ], 'int32_t')
print(trtc.Partition_Point(d_value, is_even))

d_value = trtc.device_vector_from_list([ 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 ], 'int32_t')
print(trtc.Is_Partitioned(d_value, is_even))

d_value = trtc.device_vector_from_list([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ], 'int32_t')
print(trtc.Is_Partitioned(d_value, is_even))
