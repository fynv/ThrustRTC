package JThrustRTC;

public class Kernel
{
	private static class State implements Runnable 
	{
        public long m_cptr;

        public State(long _cptr)
        {
        	m_cptr = _cptr;
        }

        @Override
        public void run() 
        {
        	Native.kernel_destroy(m_cptr);        
        }
    }

    private final State m_state;

    public long cptr() { return m_state.m_cptr; }

    public Kernel(String[] param_names, String body)
    {
    	long _cptr = Native.kernel_create(param_names, body);
		m_state = new State(_cptr);
		TRTC.cleaner.register(this, m_state);
    }

    public int num_params()
    {
        return Native.kernel_num_params(cptr());
    }

    public int calc_optimal_block_size(DeviceViewable[] args, int sharedMemBytes)
    {
    	long[] p_args = new long[args.length];
    	for (int i = 0; i<args.length; i++) p_args[i] = args[i].cptr();        
        return Native.kernel_calc_optimal_block_size(cptr(), p_args, sharedMemBytes);
    }

	public int calc_number_blocks(DeviceViewable[] args, int sizeBlock, int sharedMemBytes)
	{
		long[] p_args = new long[args.length];
    	for (int i = 0; i<args.length; i++) p_args[i] = args[i].cptr();  
		return Native.kernel_calc_number_blocks(cptr(), p_args, sizeBlock, sharedMemBytes);
	}

	public boolean launch(int[] gridDim, int[] blockDim, DeviceViewable[] args, int sharedMemBytes)
    {
        long[] p_args = new long[args.length];
    	for (int i = 0; i<args.length; i++) p_args[i] = args[i].cptr();  
        return Native.kernel_launch(cptr(), gridDim, blockDim, p_args, sharedMemBytes);
    }

    public boolean launch(int grid_x, int block_x, DeviceViewable[] args, int sharedMemBytes)
    {
        return launch(new int[]{grid_x}, new int[]{block_x}, args, sharedMemBytes);
    }

}

