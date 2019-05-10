import ThrustRTC as trtc

ctx = trtc.Context()

negate = trtc.Functor( ctx, {}, ['x'],
'''
         return -x;
''')

dinput = trtc.device_vector_from_list(ctx, [3, 7, 2, 5], 'int32_t')
doutput = trtc.device_vector(ctx, 'int32_t', 4)

dreverse = trtc.DVReverse(ctx, dinput)

trtc.Transform(ctx, dreverse, doutput, negate)
print (doutput.to_host())

