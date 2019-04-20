import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()
ctx.set_verbose()

negate = trtc.Functor( {}, ['x'], 'ret',
'''
         ret = -x;
''')

trtc.Transform(ctx, trtc.DVCounter(ctx, trtc.DVInt32(5), 10), trtc.DVDiscard(ctx, "int32_t"), negate)
