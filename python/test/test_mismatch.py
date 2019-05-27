import ThrustRTC as trtc

ctx = trtc.Context()

d1 = trtc.device_vector_from_list(ctx, [0, 5, 3, 7], 'int32_t')
d2 = trtc.device_vector_from_list(ctx, [0, 5, 8, 7], 'int32_t')

print(trtc.Mismatch(ctx, d1, d2))
print(trtc.Mismatch(ctx, d1, d2, trtc.EqualTo()))
