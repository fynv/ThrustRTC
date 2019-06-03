import ThrustRTC as trtc

ctx = trtc.Context()

dIn1 = trtc.device_vector_from_list(ctx, [ 1, 3, 5, 7, 9, 11 ], 'int32_t')
dIn2 = trtc.device_vector_from_list(ctx, [ 1, 1, 2, 3, 5, 8, 13 ], 'int32_t')
dOut = trtc.device_vector(ctx, 'int32_t', 13)

trtc.Merge(ctx, dIn1, dIn2, dOut)
print (dOut.to_host())

dIn1 = trtc.device_vector_from_list(ctx, [ 11, 9, 7, 5, 3, 1 ], 'int32_t')
dIn2 = trtc.device_vector_from_list(ctx, [ 13, 8, 5, 3, 2, 1, 1 ], 'int32_t')
dOut = trtc.device_vector(ctx, 'int32_t', 13)

trtc.Merge(ctx, dIn1, dIn2, dOut, trtc.Greater())
print (dOut.to_host())

dKeys1 = trtc.device_vector_from_list(ctx, [ 1, 3, 5, 7, 9, 11 ], 'int32_t')
dVals1 = trtc.device_vector_from_list(ctx, [ 0, 0, 0, 0, 0, 0 ], 'int32_t')
dKeys2 = trtc.device_vector_from_list(ctx, [ 1, 1, 2, 3, 5, 8, 13 ], 'int32_t')
dVals2 = trtc.device_vector_from_list(ctx, [ 1, 1, 1, 1, 1, 1, 1 ], 'int32_t')
dKeysOut = trtc.device_vector(ctx, 'int32_t', 13)
dValsOut = trtc.device_vector(ctx, 'int32_t', 13)

trtc.Merge_By_Key(ctx, dKeys1, dKeys2, dVals1, dVals2, dKeysOut, dValsOut)
print (dKeysOut.to_host())
print (dValsOut.to_host())

dKeys1 = trtc.device_vector_from_list(ctx, [ 11, 9, 7, 5, 3, 1 ], 'int32_t')
dVals1 = trtc.device_vector_from_list(ctx, [ 0, 0, 0, 0, 0, 0 ], 'int32_t')
dKeys2 = trtc.device_vector_from_list(ctx, [ 13, 8, 5, 3, 2, 1, 1 ], 'int32_t')
dVals2 = trtc.device_vector_from_list(ctx, [ 1, 1, 1, 1, 1, 1, 1 ], 'int32_t')
dKeysOut = trtc.device_vector(ctx, 'int32_t', 13)
dValsOut = trtc.device_vector(ctx, 'int32_t', 13)

trtc.Merge_By_Key(ctx, dKeys1, dKeys2, dVals1, dVals2, dKeysOut, dValsOut, trtc.Greater())
print (dKeysOut.to_host())
print (dValsOut.to_host())
