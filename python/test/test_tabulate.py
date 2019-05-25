import ThrustRTC as trtc

ctx = trtc.Context()

darr = trtc.device_vector(ctx, 'int32_t', 10)

trtc.Sequence(ctx, darr)
trtc.Tabulate(ctx, darr, trtc.Negate())
print (darr.to_host())
