import ThrustRTC as trtc

ctx = trtc.Context()

darr = trtc.device_vector_from_list(ctx, [ -1, 0, -2, -2, 1, -3], 'int32_t')

absolute_value = trtc.Functor( ctx, {}, ['x'], 
'''
         return x<(decltype(x))0 ? -x : x;
''')

print(trtc.Transform_Reduce(ctx, darr, absolute_value, trtc.DVInt32(0), trtc.Maximum()))
