import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

darr = trtc.device_vector_from_list(ctx, [ -1, 0, -2, -2, 1, -3], 'int32_t')

absolute_value = trtc.Functor( {}, ['x'], 'ret',
'''
         ret = x<(decltype(x))0 ? -x : x;
''')

maximum_value = trtc.Functor( {}, ['x', 'y'], 'ret',
'''
         ret = x<y ? y : x;
''')

print(trtc.Transform_Reduce(ctx, darr, absolute_value, trtc.DVInt32(0), maximum_value))
