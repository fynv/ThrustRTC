package JThrustRTC;

class Native 
{
	static 
	{
		System.loadLibrary("JThrustRTC_Native"); 
	}

	// Context
	public static native void set_libnvrtc_path(String path);
	public static native void set_verbose(boolean verbose);
	public static native void add_include_dir(String dir);
	public static native void add_built_in_header(String filename, String filecontent);
	public static native void add_include_filename(String filename);
	public static native void add_code_block(String code);

	public static native long kernel_create(String[] param_names, String body);
	public static native void kernel_destroy(long p_kernel);
	public static native int kernel_num_params(long p_kernel);
	public static native int kernel_calc_optimal_block_size(long p_kernel, long[] p_args, int sharedMemBytes);
	public static native int kernel_calc_number_blocks(long p_kernel, long[] p_args, int sizeBlock, int sharedMemBytes);
	public static native boolean kernel_launch(long p_kernel, int[] gridDim, int[] blockDim, long[] p_args, int sharedMemBytes);

	public static native long for_create(String[] param_names, String name_iter, String body);
	public static native void for_destroy(long p_kernel);
	public static native int for_num_params(long p_kernel);
	public static native boolean for_launch(long p_kernel, int begin, int end, long[] p_args);
	public static native boolean for_launch_n(long p_kernel, int n, long[] p_args);

	// DeviceViewable
	public static native String dv_name_view_cls(long p_dv);
	public static native void dv_destroy(long p_dv);

	public static native long dvint8_create(byte v);
	public static native byte dvint8_value(long p);
	public static native long dvint16_create(short v);
	public static native short dvint16_value(long p);
	public static native long dvint32_create(int v);
	public static native int dvint32_value(long p);
	public static native long dvint64_create(long v);
	public static native long dvint64_value(long p);
	public static native long dvfloat_create(float v);
	public static native float dvfloat_value(long p);
	public static native long dvdouble_create(double v);
	public static native double dvdouble_value(long p);

	// DVVector
	public static native String dvvectorlike_name_elem_cls(long p_dvvec);
	public static native int dvvectorlike_size(long p_dvvec);
	public static native long dvrange_create(long p_vec_value, int begin, int end);
	public static native long dvvector_create(String elem_cls, int size);
	public static native long dvvector_create(byte[] hdata);
	public static native long dvvector_create(short[] hdata);
	public static native long dvvector_create(int[] hdata);
	public static native long dvvector_create(long[] hdata);
	public static native long dvvector_create(float[] hdata);
	public static native long dvvector_create(double[] hdata);
	public static native void dvvector_to_host(long p_dvvec, byte[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, short[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, int[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, long[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, float[] hdata, int begin, int end);
	public static native void dvvector_to_host(long p_dvvec, double[] hdata, int begin, int end);
}


