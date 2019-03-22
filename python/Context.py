from . import PyThrustRTC as native

class Context:
    def __init__(self):
        self.m_cptr = native.n_context_create()

    def __del__(self):
        native.n_context_destroy(self.m_cptr)

    def set_verbose(self, verbose=True):
        native.n_context_set_verbose(self.m_cptr, verbose)

    def add_include_dir(self, path):
        native.n_context_add_include_dir(self.m_cptr, path)

    def add_built_in_header(self, filename, filecontent):
        native.n_context_add_built_in_header(self.m_cptr, filename, filecontent)

    def add_inlcude_filename(self, filename):
        native.n_context_add_inlcude_filename(self.m_cptr, filename)

    def add_preprocessor(self, line):
        native.n_context_add_preprocessor(self.m_cptr, line)

    def add_constant_object(self, name, dv):
        native.n_context_add_constant_object(self.m_cptr, name, dv.m_cptr)

    def launch_once(self, gridDim, blockDim, arg_map, code_body, sharedMemBytes=0):
        native.n_context_launch_once(
            self.m_cptr, 
            gridDim, 
            blockDim, 
            [ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()], 
            code_body, 
            sharedMemBytes)

class Kernel:
    def __init__(self, ctx, param_descs, body):
        self.m_ctx = ctx
        self.m_id = native.n_kernel_create(ctx.m_cptr, param_descs, body)

    def num_params(self):
        return native.n_kernel_num_params(self.m_ctx.m_cptr,self.m_id)

    def launch(self, gridDim, blockDim, args, sharedMemBytes=0):
        native.n_kernel_launch(self.m_ctx.m_cptr, self.m_id, gridDim, blockDim, [item.m_cptr for item in args], sharedMemBytes)

class KernelTemplate:
    def __init__(self, ctx, template_params, param_descs, body):
        self.m_ctx = ctx
        self.m_cptr = native.n_kernel_template_create(template_params, param_descs, body)

    def __del__(self):
        native.n_kernel_template_destroy(self.m_cptr)

    def num_template_params(self):
        return native.n_kernel_template_num_template_params(self.m_cptr)

    def num_params(self):
        return native.n_kernel_template_num_params(self.m_cptr)

    def launch_explict(self, gridDim, blockDim, template_args, args, sharedMemBytes=0):
        native.n_kernel_template_launch_explict(self.m_ctx.m_cptr, self.m_cptr, gridDim, blockDim, template_args, [item.m_cptr for item in args], sharedMemBytes)

    def launch(self, gridDim, blockDim, args, sharedMemBytes=0):
        native.n_kernel_template_launch(self.m_ctx.m_cptr, self.m_cptr, gridDim, blockDim, [item.m_cptr for item in args], sharedMemBytes)


