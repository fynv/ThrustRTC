import ThrustRTC as trtc



identity = trtc.Identity()

darr = trtc.device_vector_from_list([True, True, False], 'bool')

print(trtc.All_Of(darr.range(0,2), identity))
print(trtc.All_Of(darr.range(0,3), identity))
print(trtc.All_Of(darr.range(0,0), identity))

print(trtc.Any_Of(darr.range(0,2), identity))
print(trtc.Any_Of(darr.range(0,3), identity))
print(trtc.Any_Of(darr.range(2,3), identity))
print(trtc.Any_Of(darr.range(0,0), identity))

print(trtc.None_Of(darr.range(0,2), identity))
print(trtc.None_Of(darr.range(0,3), identity))
print(trtc.None_Of(darr.range(2,3), identity))
print(trtc.None_Of(darr.range(0,0), identity))
