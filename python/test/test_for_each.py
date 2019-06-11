import ThrustRTC as trtc



printf_functor =  trtc.Functor( {}, ['x'],
'''
         printf("%d\\n", x);
''')

darr = trtc.device_vector_from_list([1,2,3,1,2], 'int32_t')
trtc.For_Each(darr, printf_functor)

