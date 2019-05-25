import ThrustRTC as trtc

ctx = trtc.Context()

square_root = trtc.Functor( ctx, {}, ['x'], 
'''
         return sqrtf(x);
''')

dvalues = trtc.device_vector_from_list(ctx, [1.0, 4.0, 9.0, 16.0], 'float')
doutput = trtc.device_vector(ctx, 'float', 4)

dtrans = trtc.DVTransform(ctx, dvalues, 'float', square_root)

trtc.Transform(ctx, dtrans, doutput, trtc.Negate())
print (doutput.to_host())

