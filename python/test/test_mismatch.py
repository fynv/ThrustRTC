import ThrustRTC as trtc



d1 = trtc.device_vector_from_list([0, 5, 3, 7], 'int32_t')
d2 = trtc.device_vector_from_list([0, 5, 8, 7], 'int32_t')

print(trtc.Mismatch(d1, d2))
print(trtc.Mismatch(d1, d2, trtc.EqualTo()))
