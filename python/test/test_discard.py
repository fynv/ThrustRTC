import ThrustRTC as trtc

ctx = trtc.Context()
ctx.set_verbose()

trtc.Transform(ctx, trtc.DVCounter(ctx, trtc.DVInt32(5), 10), trtc.DVDiscard(ctx, "int32_t"), trtc.Negate())
