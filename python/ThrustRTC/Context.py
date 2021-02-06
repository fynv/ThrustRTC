from .Native import native, check_i, check_cptr
from .utils import *

def set_libnvrtc_path(path):
    native.n_set_libnvrtc_path(path.encode('utf-8'))

def Set_Verbose(verbose=True):
    native.n_set_verbose(verbose)

def Add_Include_Dir(path):
    native.n_add_include_dir(path.encode('utf-8'))

def Add_Built_In_Header(filename, filecontent):
    native.n_add_built_in_header(filename.encode('utf-8'), filecontent.encode('utf-8'))

def Add_Inlcude_Filename(filename):
    native.n_add_inlcude_filename(filename.encode('utf-8'))

def Add_Code_Block(code):
    native.n_add_code_block(code.encode('utf-8'))

def Add_Constant_Object(name, dv):
    native.n_add_constant_object(name.encode('utf-8'), dv.m_cptr)

def Wait():
    native.n_wait()

class Kernel:
    def __init__(self, param_names, body):
        o_param_names = StrArray(param_names)
        self.m_cptr = check_cptr(native.n_kernel_create(o_param_names.m_cptr, body.encode('utf-8')))

    def __del__(self):
        native.n_kernel_destroy(self.m_cptr)

    def num_params(self):
        return native.n_kernel_num_params(self.m_cptr)

    def calc_optimal_block_size(self, args, sharedMemBytes=0):
        arg_list = ObjArray(args)
        return check_i(native.n_kernel_calc_optimal_block_size(
            self.m_cptr, arg_list.m_cptr, sharedMemBytes))

    def calc_number_blocks(self, args, size_block, sharedMemBytes=0):
        arg_list = ObjArray(args)
        return check_i(native.n_kernel_calc_number_blocks(
            self.m_cptr, 
            arg_list.m_cptr, 
            size_block,
            sharedMemBytes))

    def launch(self, gridDim, blockDim, args, sharedMemBytes=0):
        d_gridDim = Dim3(gridDim)
        d_blockDim = Dim3(blockDim)
        arg_list = ObjArray(args)
        return check_i(native.n_kernel_launch(
            self.m_cptr, 
            d_gridDim.m_cptr, 
            d_blockDim.m_cptr, 
            arg_list.m_cptr, 
            sharedMemBytes))


class For:
    def __init__(self, param_names, name_iter, body):
        o_param_names = StrArray(param_names)
        self.m_cptr = check_cptr(native.n_for_create(o_param_names.m_cptr, name_iter.encode('utf-8'), body.encode('utf-8')))

    def __del__(self):
        native.n_for_destroy(self.m_cptr)

    def num_params(self):
        return native.n_for_num_params(self.m_cptr)

    def launch(self, begin, end, args):
        arg_list = ObjArray(args)
        check_i(native.n_for_launch(self.m_cptr, begin, end, arg_list.m_cptr))

    def launch_n(self, n, args):
        arg_list = ObjArray(args)
        check_i(native.n_for_launch_n(self.m_cptr, n, arg_list.m_cptr))
