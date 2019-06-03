import ThrustRTC as trtc

ctx = trtc.Context()

d_values = trtc.device_vector_from_list(ctx, [0, 2, 5, 7, 8], 'int32_t')

print(trtc.Lower_Bound(ctx, d_values, trtc.DVInt32(0)))
print(trtc.Lower_Bound(ctx, d_values, trtc.DVInt32(1)))
print(trtc.Lower_Bound(ctx, d_values, trtc.DVInt32(2)))
print(trtc.Lower_Bound(ctx, d_values, trtc.DVInt32(3)))
print(trtc.Lower_Bound(ctx, d_values, trtc.DVInt32(8)))
print(trtc.Lower_Bound(ctx, d_values, trtc.DVInt32(9)))

print()

print(trtc.Upper_Bound(ctx, d_values, trtc.DVInt32(0)))
print(trtc.Upper_Bound(ctx, d_values, trtc.DVInt32(1)))
print(trtc.Upper_Bound(ctx, d_values, trtc.DVInt32(2)))
print(trtc.Upper_Bound(ctx, d_values, trtc.DVInt32(3)))
print(trtc.Upper_Bound(ctx, d_values, trtc.DVInt32(8)))
print(trtc.Upper_Bound(ctx, d_values, trtc.DVInt32(9)))

print()

print(trtc.Binary_Search(ctx, d_values, trtc.DVInt32(0)))
print(trtc.Binary_Search(ctx, d_values, trtc.DVInt32(1)))
print(trtc.Binary_Search(ctx, d_values, trtc.DVInt32(2)))
print(trtc.Binary_Search(ctx, d_values, trtc.DVInt32(3)))
print(trtc.Binary_Search(ctx, d_values, trtc.DVInt32(8)))
print(trtc.Binary_Search(ctx, d_values, trtc.DVInt32(9)))
