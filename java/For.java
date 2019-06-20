package JThrustRTC;

public class For
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
        	Native.for_destroy(m_cptr);        
        }
    }

    private final State m_state;

    public long cptr() { return m_state.m_cptr; }

	public For(String[] param_names, String name_iter, String body)
    {
    	long _cptr = Native.for_create(param_names, name_iter, body);
		m_state = new State(_cptr);
		TRTC.cleaner.register(this, m_state);
    }

    public int num_params()
    {
        return Native.for_num_params(cptr());
    }

	public boolean launch(int begin, int end, DeviceViewable[] args)
	{
		long[] p_args = new long[args.length];
    	for (int i = 0; i<args.length; i++) p_args[i] = args[i].cptr();  
		return Native.for_launch(cptr(), begin, end, p_args);
	}

	public boolean launch_n(int n, DeviceViewable[] args)
	{
		long[] p_args = new long[args.length];
    	for (int i = 0; i<args.length; i++) p_args[i] = args[i].cptr();  
		return Native.for_launch_n(cptr(), n, p_args);
	}
}