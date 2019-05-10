import ThrustRTC as trtc

ctx = trtc.Context()

dIn = trtc.device_vector_from_list(ctx, [ 10, 20, 30, 40, 50, 60, 70, 80], 'int32_t')
dOut = trtc.device_vector(ctx, 'int32_t', 8)

trtc.Copy(ctx, dIn, dOut)
print (dOut.to_host())
