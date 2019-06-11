import ThrustRTC as trtc



dIn = trtc.device_vector_from_list([ 10, 20, 30, 40, 50, 60, 70, 80 ], 'int32_t')
dOut = trtc.device_vector('int32_t', 8)

trtc.Copy(dIn, dOut)
print (dOut.to_host())


is_even = trtc.Functor( {}, ['x'], 
'''
         return x % 2 == 0;
''')

dIn = trtc.device_vector_from_list([ -2, 0, -1, 0, 1, 2 ], 'int32_t')
dOut = trtc.device_vector('int32_t', 6)
count = trtc.Copy_If(dIn, dOut, is_even)
print (dOut.to_host(0, count))

dIn = trtc.device_vector_from_list([ 0, 1, 2, 3, 4, 5 ], 'int32_t')
dStencil = trtc.device_vector_from_list([ -2, 0, -1, 0, 1, 2 ], 'int32_t')
dOut = trtc.device_vector('int32_t', 6)
count = trtc.Copy_If_Stencil(dIn, dStencil, dOut, is_even)
print (dOut.to_host(0, count))
