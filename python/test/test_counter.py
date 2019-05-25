import ThrustRTC as trtc

ctx = trtc.Context()

negate = trtc.Functor( ctx, {}, ['x'],
'''
         return -x;
''')


darr = trtc.device_vector(ctx, 'int32_t', 10)
trtc.Transform(ctx, trtc.DVCounter(ctx, trtc.DVInt32(5), 10), darr, trtc.Negate())
print (darr.to_host())
