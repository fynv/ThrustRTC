from . import PyThrustRTC as native
from .Context import *

class For:
    def __init__(self, ctx, param_descs, name_iter, body):
    	self.m_cptr = native.n_for_create(ctx.m_cptr, param_descs, name_iter, body)

    def __del__(self):
        native.n_for_destroy(self.m_cptr)

    def num_params(self):
        return native.n_for_num_params(self.m_cptr)

    def launch(self, begin, end, args, sharedMemBytes=0):
        native.n_for_launch(self.m_cptr, begin, end, [item.m_cptr for item in args], sharedMemBytes)

class ForTemplate:
    def __init__(self, ctx, template_params, param_descs, name_iter, body):
        self.m_ctx = ctx
        self.m_cptr = native.n_for_template_create(template_params, param_descs, name_iter, body)

    def __del__(self):
        native.n_for_template_destroy(self.m_cptr)

    def num_template_params(self):
        return native.n_for_template_num_template_params(self.m_cptr)

    def num_params(self):
        return native.n_for_template_num_params(self.m_cptr)

    def launch_explict(self, begin, end, template_args, args, sharedMemBytes=0):
        native.n_for_template_launch_explict(self.m_ctx.m_cptr, self.m_cptr, begin, end, template_args, [item.m_cptr for item in args], sharedMemBytes)

    def launch(self, begin, end, args, sharedMemBytes=0):
        native.n_for_template_launch(self.m_ctx.m_cptr, self.m_cptr, begin, end, [item.m_cptr for item in args], sharedMemBytes)

def ForOnce(ctx, begin, end, arg_map, name_iter, code_body, sharedMemBytes=0):
    native.n_for_launch_once(
        ctx.m_cptr, 
        begin, 
        end, 
        [ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()],
        name_iter, 
        code_body, 
        sharedMemBytes)
