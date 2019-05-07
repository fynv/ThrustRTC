import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

negate = trtc.Functor( ctx, {}, ['x'],
'''
         return -x;
''')

square_root = trtc.Functor( ctx, {}, ['x'], 
'''
         return sqrtf(x);
''')

dvalues = trtc.device_vector_from_list(ctx, [1.0, 4.0, 9.0, 16.0], 'float')
doutput = trtc.device_vector(ctx, 'float', 4)

dtrans = trtc.DVTransform(ctx, dvalues, 'float', square_root)

trtc.Transform(ctx, dtrans, doutput, negate)
print (doutput.to_host())

