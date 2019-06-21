package JThrustRTC;

public class DVTuple extends DeviceViewable
{
	private static long create(DeviceViewable[] objs, String[] name_objs)
	{		
		long[] p_objs = new long[objs.length];
		for (int i = 0; i<objs.length; i++ )
			p_objs[i] = objs[i].cptr();
		return Native.dvtuple_create(p_objs, name_objs);
	}

	public DVTuple(DeviceViewable[] objs, String[] name_objs)
	{
		super(create(objs, name_objs));
		m_objs = objs;
	}

    private DeviceViewable[] m_objs;

    @Override
    public void close()
    {
    	m_objs = null;
        super.close();
    }
}
