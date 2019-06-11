import ThrustRTC as trtc



negate = trtc.Functor( {}, ['x'],
'''
         return -x;
''')


darr = trtc.device_vector('int32_t', 10)
trtc.Transform(trtc.DVCounter(trtc.DVInt32(5), 10), darr, trtc.Negate())
print (darr.to_host())
