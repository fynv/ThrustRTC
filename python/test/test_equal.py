import ThrustRTC as trtc



compare_modulo_two = trtc.Functor( {}, ['x','y'],
'''
         return (x % 2) == (y % 2);
''')

darr1 = trtc.device_vector_from_list([ 3, 1, 4, 1, 5, 9, 3], 'int32_t')
darr2 = trtc.device_vector_from_list([ 3, 1, 4, 2, 8, 5, 7], 'int32_t')
darr3 = trtc.device_vector_from_list([ 3, 1, 4, 1, 5, 9, 3], 'int32_t')
print(trtc.Equal(darr1, darr2))
print(trtc.Equal(darr1, darr3))

dx = trtc.device_vector_from_list([ 1, 2, 3, 4, 5, 6 ], 'int32_t')
dy = trtc.device_vector_from_list([ 7, 8, 9, 10, 11, 12 ], 'int32_t')
print(trtc.Equal(dx, dy, compare_modulo_two))
