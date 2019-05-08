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

    def add_code_block(self, code):
        native.n_context_add_code_block(self.m_cptr, code)

    def add_constant_object(self, name, dv):
        native.n_context_add_constant_object(self.m_cptr, name, dv.m_cptr)

    def calc_optimal_block_size(self, arg_map, code_body, sharedMemBytes=0):
        return native.n_context_calc_optimal_block_size(
            self.m_cptr,
            [ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()], 
            code_body, 
            sharedMemBytes)

    def launch_kernel(self, gridDim, blockDim, arg_map, code_body, sharedMemBytes=0):
        native.n_context_launch_kernel(
            self.m_cptr, 
            gridDim, 
            blockDim, 
            [ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()], 
            code_body, 
            sharedMemBytes)

    def launch_for(self, begin, end, arg_map, name_iter, code_body):
        native.n_context_launch_for(
            self.m_cptr, 
            begin, 
            end, 
            [ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()], 
            name_iter,
            code_body)

    def launch_for_n(self, n, arg_map, name_iter, code_body):
        native.n_context_launch_for_n(
            self.m_cptr, 
            n, 
            [ (param_name, arg.m_cptr) for param_name, arg in arg_map.items()], 
            name_iter,
            code_body)

class Kernel:
    def __init__(self, param_descs, body):
        self.m_cptr = native.n_kernel_create(param_descs, body)

    def __del__(self):
        native.n_kernel_destroy(self.m_cptr)

    def num_params(self):
        return native.n_kernel_num_params(self.m_cptr)

    def calc_optimal_block_size(self, ctx, args, sharedMemBytes=0):
        return native.n_kernel_calc_optimal_block_size(
            ctx.m_cptr,
            self.m_cptr, 
            [item.m_cptr for item in args], 
            sharedMemBytes)

    def launch(self, ctx, gridDim, blockDim, args, sharedMemBytes=0):
        native.n_kernel_launch(
            ctx.m_cptr, 
            self.m_cptr, 
            gridDim, 
            blockDim, 
            [item.m_cptr for item in args], 
            sharedMemBytes)

class For:
    def __init__(self, param_descs, name_iter, body):
        self.m_cptr = native.n_for_create(param_descs, name_iter, body)

    def __del__(self):
        native.n_for_destroy(self.m_cptr)

    def num_params(self):
        return native.n_for_num_params(self.m_cptr)

    def launch(self, ctx, begin, end, args):
        native.n_for_launch(ctx.m_cptr, self.m_cptr, begin, end, [item.m_cptr for item in args])

    def launch_n(self, ctx, n, args):
        native.n_for_launch_n(ctx.m_cptr, self.m_cptr, n, [item.m_cptr for item in args])
