import ThrustRTC as trtc

ctx = trtc.Context()

dvalues = trtc.device_vector_from_list(ctx, [ 1, 4, 2, 8, 5, 7 ], 'int32_t')
trtc.Sort(ctx, dvalues)
print (dvalues.to_host())

dvalues = trtc.device_vector_from_list(ctx, [ 1, 4, 2, 8, 5, 7 ], 'int32_t')
trtc.Sort(ctx, dvalues, trtc.Greater())
print (dvalues.to_host())

dkeys = trtc.device_vector_from_list(ctx, [ 1, 4, 2, 8, 5, 7 ], 'int32_t')
dvalues = trtc.device_vector_from_list(ctx, [ 1, 2, 3, 4, 5, 6], 'int32_t')
trtc.Sort_By_Key(ctx, dkeys, dvalues)
print (dkeys.to_host())
print (dvalues.to_host())

dkeys = trtc.device_vector_from_list(ctx, [ 1, 4, 2, 8, 5, 7 ], 'int32_t')
dvalues = trtc.device_vector_from_list(ctx, [ 1, 2, 3, 4, 5, 6], 'int32_t')
trtc.Sort_By_Key(ctx, dkeys, dvalues, trtc.Greater())
print (dkeys.to_host())
print (dvalues.to_host())
