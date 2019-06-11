import ThrustRTC as trtc



d_values = trtc.device_vector_from_list([0, 5, 3, 7], 'int32_t')
print(trtc.Find(d_values, trtc.DVInt32(3)))
print(trtc.Find(d_values,trtc.DVInt32(5)))
print(trtc.Find(d_values,trtc.DVInt32(9)))
print(trtc.Find_If(d_values, trtc.Functor({}, ['x'], '        return x>4;\n')))
print(trtc.Find_If(d_values, trtc.Functor({}, ['x'], '        return x>10;\n')))
print(trtc.Find_If_Not(d_values, trtc.Functor({}, ['x'], '        return x>4;\n')))
print(trtc.Find_If_Not(d_values, trtc.Functor({}, ['x'], '        return x>10;\n')))
