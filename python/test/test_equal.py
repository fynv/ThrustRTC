import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

compare_modulo_two = trtc.Functor( ctx, {}, ['x','y'],
'''
         return (x % 2) == (y % 2);
''')

darr1 = trtc.device_vector_from_list(ctx, [ 3, 1, 4, 1, 5, 9, 3], 'int32_t')
darr2 = trtc.device_vector_from_list(ctx, [ 3, 1, 4, 2, 8, 5, 7], 'int32_t')
darr3 = trtc.device_vector_from_list(ctx, [ 3, 1, 4, 1, 5, 9, 3], 'int32_t')
print(trtc.Equal(ctx, darr1, darr2))
print(trtc.Equal(ctx, darr1, darr3))

dx = trtc.device_vector_from_list(ctx, [ 1, 2, 3, 4, 5, 6 ], 'int32_t')
dy = trtc.device_vector_from_list(ctx, [ 7, 8, 9, 10, 11, 12 ], 'int32_t')
print(trtc.Equal(ctx, dx, dy, compare_modulo_two))
