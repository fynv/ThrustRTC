import ThrustRTC as trtc

ctx = trtc.Context()

negate = trtc.Functor( ctx, {}, ['x'], 
'''
         return -x;
''')


darr = trtc.device_vector(ctx, 'int32_t', 10)

trtc.Sequence(ctx, darr)
trtc.Tabulate(ctx, darr, negate)
print (darr.to_host())
