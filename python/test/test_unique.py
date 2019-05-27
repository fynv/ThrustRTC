import ThrustRTC as trtc

ctx = trtc.Context()

d_value = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
count = trtc.Unique(ctx, d_value)
print (d_value.to_host(0, count))

d_value = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
count = trtc.Unique(ctx, d_value, trtc.EqualTo())
print (d_value.to_host(0, count))

d_in = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
d_out = trtc.device_vector(ctx, 'int32_t', 7)
count = trtc.Unique_Copy(ctx, d_in, d_out)
print (d_out.to_host(0, count))

d_in = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
d_out = trtc.device_vector(ctx, 'int32_t', 7)
count = trtc.Unique_Copy(ctx, d_in, d_out, trtc.EqualTo())
print (d_out.to_host(0, count))

d_keys = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
d_values =  trtc.device_vector_from_list(ctx, [ 9, 8, 7, 6, 5, 4, 3], 'int32_t')
count = trtc.Unique_By_Key(ctx, d_keys, d_values)
print (d_keys.to_host(0, count))
print (d_values.to_host(0, count))

d_keys = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
d_values =  trtc.device_vector_from_list(ctx, [ 9, 8, 7, 6, 5, 4, 3], 'int32_t')
count = trtc.Unique_By_Key(ctx, d_keys, d_values, trtc.EqualTo())
print (d_keys.to_host(0, count))
print (d_values.to_host(0, count))

d_keys_in = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
d_values_in =  trtc.device_vector_from_list(ctx, [ 9, 8, 7, 6, 5, 4, 3], 'int32_t')
d_keys_out = trtc.device_vector(ctx, 'int32_t', 7)
d_values_out = trtc.device_vector(ctx, 'int32_t', 7)
count = trtc.Unique_By_Key_Copy(ctx, d_keys_in, d_values_in, d_keys_out, d_values_out)
print (d_keys_out.to_host(0, count))
print (d_values_out.to_host(0, count))

d_keys_in = trtc.device_vector_from_list(ctx, [ 1, 3, 3, 3, 2, 2, 1 ], 'int32_t')
d_values_in =  trtc.device_vector_from_list(ctx, [ 9, 8, 7, 6, 5, 4, 3], 'int32_t')
d_keys_out = trtc.device_vector(ctx, 'int32_t', 7)
d_values_out = trtc.device_vector(ctx, 'int32_t', 7)
count = trtc.Unique_By_Key_Copy(ctx, d_keys_in, d_values_in, d_keys_out, d_values_out, trtc.EqualTo())
print (d_keys_out.to_host(0, count))
print (d_values_out.to_host(0, count))
