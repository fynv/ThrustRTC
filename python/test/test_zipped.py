import ThrustRTC as trtc



d_int_in = trtc.device_vector_from_list([0, 1, 2, 3, 4], 'int32_t')
d_float_in = trtc.device_vector_from_list([ 0.0, 10.0, 20.0, 30.0, 40.0], 'float')

d_int_out = trtc.device_vector('int32_t', 5)
d_float_out = trtc.device_vector('float', 5)

zipped_in = trtc.DVZipped([d_int_in, d_float_in], ['a','b'])
zipped_out = trtc.DVZipped([d_int_out, d_float_out], ['a','b'])

trtc.Copy(zipped_in, zipped_out)
print (d_int_out.to_host())
print (d_float_out.to_host())

d_int_in = trtc.DVCounter(trtc.DVInt32(0), 5)
d_float_in = trtc.DVTransform(d_int_in, "float", trtc.Functor({}, ['i'], '        return (float)i*10.0f +10.0f;\n'))
zipped_in = trtc.DVZipped([d_int_in, d_float_in], ['a','b'])
trtc.Copy(zipped_in, zipped_out)
print (d_int_out.to_host())
print (d_float_out.to_host())

const_in = trtc.DVConstant(trtc.DVTuple({'a': trtc.DVInt32(123), 'b': trtc.DVFloat(456.0)}), 5)
trtc.Copy(const_in, zipped_out)
print (d_int_out.to_host())
print (d_float_out.to_host())
