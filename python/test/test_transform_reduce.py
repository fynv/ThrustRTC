import ThrustRTC as trtc



darr = trtc.device_vector_from_list([ -1, 0, -2, -2, 1, -3], 'int32_t')

absolute_value = trtc.Functor( {}, ['x'], 
'''
         return x<(decltype(x))0 ? -x : x;
''')

print(trtc.Transform_Reduce(darr, absolute_value, trtc.DVInt32(0), trtc.Maximum()))
