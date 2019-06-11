import ThrustRTC as trtc



d_input = trtc.device_vector_from_list([0, 2, 5, 7, 8], 'int32_t')

print(trtc.Lower_Bound(d_input, trtc.DVInt32(0)))
print(trtc.Lower_Bound(d_input, trtc.DVInt32(1)))
print(trtc.Lower_Bound(d_input, trtc.DVInt32(2)))
print(trtc.Lower_Bound(d_input, trtc.DVInt32(3)))
print(trtc.Lower_Bound(d_input, trtc.DVInt32(8)))
print(trtc.Lower_Bound(d_input, trtc.DVInt32(9)))

print()

print(trtc.Upper_Bound(d_input, trtc.DVInt32(0)))
print(trtc.Upper_Bound(d_input, trtc.DVInt32(1)))
print(trtc.Upper_Bound(d_input, trtc.DVInt32(2)))
print(trtc.Upper_Bound(d_input, trtc.DVInt32(3)))
print(trtc.Upper_Bound(d_input, trtc.DVInt32(8)))
print(trtc.Upper_Bound(d_input, trtc.DVInt32(9)))

print()

print(trtc.Binary_Search(d_input, trtc.DVInt32(0)))
print(trtc.Binary_Search(d_input, trtc.DVInt32(1)))
print(trtc.Binary_Search(d_input, trtc.DVInt32(2)))
print(trtc.Binary_Search(d_input, trtc.DVInt32(3)))
print(trtc.Binary_Search(d_input, trtc.DVInt32(8)))
print(trtc.Binary_Search(d_input, trtc.DVInt32(9)))

print()

d_values = trtc.device_vector_from_list([0, 1, 2, 3, 8, 9], 'int32_t')
d_output = trtc.device_vector('int32_t', 6)

trtc.Lower_Bound_V(d_input, d_values, d_output)
print(d_output.to_host())

trtc.Upper_Bound_V(d_input, d_values, d_output)
print(d_output.to_host())

trtc.Binary_Search_V(d_input, d_values, d_output)
print(d_output.to_host())
