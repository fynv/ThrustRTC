import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

is_even = trtc.Functor( {}, ['x'], 'ret',
'''
         ret = ((x % 2) == 0);
''')

dvalues = trtc.device_vector_from_list(ctx, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 'int32_t')
dmap = trtc.device_vector_from_list(ctx, [ 0, 2, 4, 6, 8, 1, 3, 5, 7, 9], 'int32_t')
doutput = trtc.device_vector(ctx, 'int32_t', 10)

trtc.Gather(ctx, dmap, dvalues, doutput)
print (doutput.to_host())

dvalues = trtc.device_vector_from_list(ctx, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], 'int32_t')
dstencil = trtc.device_vector_from_list(ctx, [ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ], 'int32_t')
dmap = trtc.device_vector_from_list(ctx, [ 0, 2, 4, 6, 8, 1, 3, 5, 7, 9], 'int32_t')
doutput = trtc.device_vector_from_list(ctx, [ 7,7,7,7,7,7,7,7,7,7], 'int32_t')

trtc.Gather_If(ctx, dmap, dstencil, dvalues, doutput)
print (doutput.to_host())


dvalues = trtc.device_vector_from_list(ctx, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], 'int32_t')
dstencil = trtc.device_vector_from_list(ctx, [ 0, 3, 4, 1, 4, 1, 2, 7, 8, 9 ], 'int32_t')
dmap = trtc.device_vector_from_list(ctx, [ 0, 2, 4, 6, 8, 1, 3, 5, 7, 9], 'int32_t')
doutput = trtc.device_vector_from_list(ctx, [ 7,7,7,7,7,7,7,7,7,7], 'int32_t')

trtc.Gather_If(ctx, dmap, dstencil, dvalues, doutput, is_even)
print (doutput.to_host())


