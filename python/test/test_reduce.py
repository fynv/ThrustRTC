import ThrustRTC as trtc

ctx = trtc.Context()

darr = trtc.device_vector_from_list(ctx, [1, 0, 2, 2, 1, 3], 'int32_t')
print(trtc.Reduce(ctx, darr))
print(trtc.Reduce(ctx, darr, trtc.DVInt32(1)))
print(trtc.Reduce(ctx, darr, trtc.DVInt32(-1), trtc.Maximum()))

d_keys_in = trtc.device_vector_from_list(ctx, [1, 3, 3, 3, 2, 2, 1], 'int32_t')
d_value_in = trtc.device_vector_from_list(ctx, [9, 8, 7, 6, 5, 4, 3], 'int32_t')

d_keys_out = trtc.device_vector(ctx, 'int32_t', 7)
d_values_out = trtc.device_vector(ctx, 'int32_t', 7)

count = trtc.Reduce_By_Key(ctx, d_keys_in, d_value_in, d_keys_out, d_values_out)
print (d_keys_out.to_host(0, count))
print (d_values_out.to_host(0, count))

count = trtc.Reduce_By_Key(ctx, d_keys_in, d_value_in, d_keys_out, d_values_out, trtc.EqualTo())
print (d_keys_out.to_host(0, count))
print (d_values_out.to_host(0, count))

count = trtc.Reduce_By_Key(ctx, d_keys_in, d_value_in, d_keys_out, d_values_out, trtc.EqualTo(), trtc.Plus())
print (d_keys_out.to_host(0, count))
print (d_values_out.to_host(0, count))
