import ThrustRTC as trtc

ctx = trtc.Context()

is_odd = trtc.Functor( ctx, {}, ['x'], 
'''
         return x % 2;
''')

darr = trtc.device_vector_from_list(ctx, [-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], 'int32_t')
trtc.Transform(ctx, darr, darr, trtc.Negate())
print (darr.to_host())

darr_in1 = trtc.device_vector_from_list(ctx, [-5,  0,  2,  3,  2,  4 ], 'int32_t')
darr_in2 = trtc.device_vector_from_list(ctx, [ 3,  6, -2,  1,  2,  3 ], 'int32_t')
darr_out = trtc.device_vector(ctx, 'int32_t', 6)
trtc.Transform_Binary(ctx, darr_in1, darr_in2, darr_out, trtc.Plus())
print (darr_out.to_host())

darr = trtc.device_vector_from_list(ctx, [-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], 'int32_t')
trtc.Transform_If(ctx, darr, darr, trtc.Negate(), is_odd)
print (darr.to_host())

darr_data = trtc.device_vector_from_list(ctx, [-5, 0, 2, -3, 2, 4, 0, -1, 2, 8 ], 'int32_t')
darr_stencil = trtc.device_vector_from_list(ctx, [1, 0, 1,  0, 1, 0, 1,  0, 1, 0], 'int32_t')
trtc.Transform_If_Stencil(ctx, darr_data, darr_stencil, darr_data, trtc.Negate(), trtc.Identity())
print (darr_data.to_host())

darr_in1 = trtc.device_vector_from_list(ctx, [-5,  0,  2,  3,  2,  4 ], 'int32_t')
darr_in2 = trtc.device_vector_from_list(ctx, [ 3,  6, -2,  1,  2,  3 ], 'int32_t')
darr_stencil = trtc.device_vector_from_list(ctx, [ 1,  0,  1,  0,  1,  0], 'int32_t')
darr_out =  trtc.device_vector_from_list(ctx, [ -1, -1, -1, -1, -1, -1 ], 'int32_t')
trtc.Transform_Binary_If_Stencil(ctx, darr_in1, darr_in2, darr_stencil, darr_out, trtc.Plus(), trtc.Identity())
print (darr_out.to_host())
