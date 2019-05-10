import ThrustRTC as trtc

ctx = trtc.Context()

is_even = trtc.Functor( ctx, {}, ['x'], 
'''
         return ((x % 2) == 0);
''')

dvalues = trtc.device_vector_from_list(ctx, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 'int32_t')
dmap = trtc.device_vector_from_list(ctx, [ 0, 5, 1, 6, 2, 7, 3, 8, 4, 9], 'int32_t')
doutput = trtc.device_vector(ctx, 'int32_t', 10)

trtc.Scatter(ctx, dvalues, dmap, doutput)
print (doutput.to_host())

V = trtc.device_vector_from_list(ctx, [10, 20, 30, 40, 50, 60, 70, 80], 'int32_t')
M = trtc.device_vector_from_list(ctx, [ 0, 5, 1, 6, 2, 7, 3, 4], 'int32_t')
S = trtc.device_vector_from_list(ctx, [ 1, 0, 1, 0, 1, 0, 1, 0], 'int32_t')
D = trtc.device_vector_from_list(ctx, [ 0, 0, 0, 0, 0, 0, 0, 0], 'int32_t')

trtc.Scatter_If(ctx, V, M, S, D)
print (D.to_host())

V = trtc.device_vector_from_list(ctx, [10, 20, 30, 40, 50, 60, 70, 80], 'int32_t')
M = trtc.device_vector_from_list(ctx, [ 0, 5, 1, 6, 2, 7, 3, 4], 'int32_t')
S = trtc.device_vector_from_list(ctx, [ 2, 1, 2, 1, 2, 1, 2, 1], 'int32_t')
D = trtc.device_vector_from_list(ctx, [ 0, 0, 0, 0, 0, 0, 0, 0], 'int32_t')

trtc.Scatter_If(ctx, V, M, S, D, is_even)
print (D.to_host())
