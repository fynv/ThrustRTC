import PyThrustRTC as native

def Set_Verbose(verbose=True):
    native.n_set_verbose(verbose)

def Add_Include_Dir(path):
    native.n_add_include_dir(path)

def Add_Built_In_Header(filename, filecontent):
    native.n_add_built_in_header(filename, filecontent)

def Add_Inlcude_Filename(filename):
    native.n_add_inlcude_filename(filename)

def Add_Code_Block(code):
    native.n_add_code_block(code)

def Add_Constant_Object(name, dv):
    native.n_add_constant_object(name, dv.m_cptr)

class Kernel:
    def __init__(self, param_names, body):
        self.m_cptr = native.n_kernel_create(param_names, body)

    def __del__(self):
        native.n_kernel_destroy(self.m_cptr)

    def num_params(self):
        return native.n_kernel_num_params(self.m_cptr)

    def calc_optimal_block_size(self, args, sharedMemBytes=0):
        return native.n_kernel_calc_optimal_block_size(
            self.m_cptr, 
            [item.m_cptr for item in args], 
            sharedMemBytes)

    def calc_number_blocks(self, args, size_block, sharedMemBytes=0):
        return native.n_kernel_calc_number_blocks(
            self.m_cptr, 
            [item.m_cptr for item in args], 
            size_block,
            sharedMemBytes)

    def launch(self, gridDim, blockDim, args, sharedMemBytes=0):
        native.n_kernel_launch(
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

    def launch(self, begin, end, args):
        native.n_for_launch(self.m_cptr, begin, end, [item.m_cptr for item in args])

    def launch_n(self, n, args):
        native.n_for_launch_n(self.m_cptr, n, [item.m_cptr for item in args])
