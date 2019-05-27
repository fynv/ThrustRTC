import ThrustRTC as trtc

ctx = trtc.Context()

d_values = trtc.device_vector_from_list(ctx, [0, 5, 3, 7], 'int32_t')
print(trtc.Find(ctx, d_values, trtc.DVInt32(3)))
print(trtc.Find(ctx, d_values,trtc.DVInt32(5)))
print(trtc.Find(ctx, d_values,trtc.DVInt32(9)))
print(trtc.Find_If(ctx, d_values, trtc.Functor(ctx, {}, ['x'], '        return x>4;\n')))
print(trtc.Find_If(ctx, d_values, trtc.Functor(ctx, {}, ['x'], '        return x>10;\n')))
print(trtc.Find_If_Not(ctx, d_values, trtc.Functor(ctx, {}, ['x'], '        return x>4;\n')))
print(trtc.Find_If_Not(ctx, d_values, trtc.Functor(ctx, {}, ['x'], '        return x>10;\n')))
