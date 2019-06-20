package JThrustRTC;

public class DeviceViewable
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
        	Native.dv_destroy(m_cptr);        
        }
    }

	private final State m_state;

	public long cptr() { return m_state.m_cptr; }

	public String name_view_cls()
	{
		return Native.dv_name_view_cls(m_state.m_cptr);
	}

	public DeviceViewable(long _cptr)
    {
         m_state = new State(_cptr);
         TRTC.cleaner.register(this, m_state);
    }

} 

