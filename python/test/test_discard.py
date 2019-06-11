import ThrustRTC as trtc

trtc.Set_Verbose()

trtc.Transform(trtc.DVCounter(trtc.DVInt32(5), 10), trtc.DVDiscard("int32_t"), trtc.Negate())
