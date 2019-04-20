import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

negate = trtc.Functor( {}, ['x'], 'ret',
'''
         ret = -x;
''')

dvalues = trtc.device_vector_from_list(ctx, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 'float')
dindices =  trtc.device_vector_from_list(ctx, [2,6,1,3], 'int32_t')
doutput = trtc.device_vector(ctx, 'float', 4)

perm = trtc.DVPermutation(ctx, dvalues, dindices)

trtc.Transform(ctx, perm, doutput, negate)
print (doutput.to_host())

