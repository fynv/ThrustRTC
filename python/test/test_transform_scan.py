import ThrustRTC as trtc

ctx = trtc.Context()

negate = trtc.Functor( ctx, {}, ['x'], 
'''
         return -x;
''')

plus = trtc.Functor( ctx, {}, ['x', 'y'], 
'''
         return x+y;
''')

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Transform_Inclusive_Scan(ctx, darr, darr, negate, plus)
print (darr.to_host())

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
trtc.Transform_Exclusive_Scan(ctx, darr, darr, negate, trtc.DVInt32(4), plus)
print (darr.to_host())

