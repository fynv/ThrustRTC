import ThrustRTC as trtc



is_even = trtc.Functor( {}, ['x'], 
'''
         return ((x % 2) == 0);
''')

dvalues = trtc.device_vector_from_list([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 'int32_t')
dmap = trtc.device_vector_from_list([ 0, 2, 4, 6, 8, 1, 3, 5, 7, 9], 'int32_t')
doutput = trtc.device_vector('int32_t', 10)

trtc.Gather(dmap, dvalues, doutput)
print (doutput.to_host())

dvalues = trtc.device_vector_from_list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], 'int32_t')
dstencil = trtc.device_vector_from_list([ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ], 'int32_t')
dmap = trtc.device_vector_from_list([ 0, 2, 4, 6, 8, 1, 3, 5, 7, 9], 'int32_t')
doutput = trtc.device_vector_from_list([ 7,7,7,7,7,7,7,7,7,7], 'int32_t')

trtc.Gather_If(dmap, dstencil, dvalues, doutput)
print (doutput.to_host())


dvalues = trtc.device_vector_from_list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], 'int32_t')
dstencil = trtc.device_vector_from_list([ 0, 3, 4, 1, 4, 1, 2, 7, 8, 9 ], 'int32_t')
dmap = trtc.device_vector_from_list([ 0, 2, 4, 6, 8, 1, 3, 5, 7, 9], 'int32_t')
doutput = trtc.device_vector_from_list([ 7,7,7,7,7,7,7,7,7,7], 'int32_t')

trtc.Gather_If(dmap, dstencil, dvalues, doutput, is_even)
print (doutput.to_host())


