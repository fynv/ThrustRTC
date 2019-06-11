import ThrustRTC as trtc



op = trtc.Functor( {}, ['x'], 
'''
         return x % 100;
''')


darr = trtc.device_vector('int32_t', 2000)
trtc.Transform(trtc.DVCounter(trtc.DVInt32(0), 2000), darr, op)
print(trtc.Count(darr, trtc.DVInt32(47)))


op2 = trtc.Functor({}, ['x'],
'''
         return (x % 100)==47;
''')

trtc.Sequence(darr)
print(trtc.Count_If(darr, op2))
