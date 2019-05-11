import ThrustRTC as trtc

ctx = trtc.Context()

d_int_in = trtc.device_vector_from_list(ctx, [0, 1, 2, 3, 4], 'int32_t')
d_float_in = trtc.device_vector_from_list(ctx, [ 0.0, 10.0, 20.0, 30.0, 40.0], 'float')

d_int_out = trtc.device_vector(ctx, 'int32_t', 5)
d_float_out = trtc.device_vector(ctx, 'float', 5)

zipped_in = trtc.DVZipped(ctx, [d_int_in, d_float_in], ['a','b'])
zipped_out = trtc.DVZipped(ctx, [d_int_out, d_float_out], ['a','b'])

trtc.Copy(ctx, zipped_in, zipped_out)
print (d_int_out.to_host())
print (d_float_out.to_host())

d_int_in = trtc.DVCounter(ctx, trtc.DVInt32(0), 5)
d_float_in = trtc.DVTransform(ctx, d_int_in, "float", trtc.Functor(ctx, {}, ['i'], '        return (float)i*10.0f +10.0f;\n'))
zipped_in = trtc.DVZipped(ctx, [d_int_in, d_float_in], ['a','b'])
trtc.Copy(ctx, zipped_in, zipped_out)
print (d_int_out.to_host())
print (d_float_out.to_host())

const_in = trtc.DVConstant(ctx, trtc.DVTuple(ctx, {'a': trtc.DVInt32(123), 'b': trtc.DVFloat(456.0)}), 5)
trtc.Copy(ctx, const_in, zipped_out)
print (d_int_out.to_host())
print (d_float_out.to_host())
