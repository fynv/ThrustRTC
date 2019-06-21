package JThrustRTC;

public class Functor extends DeviceViewable
{
	public Functor(String[] functor_params, String code_body)
	{
		super(Native.functor_create(functor_params, code_body));
		m_objs = null;
	}

	private static long create(DeviceViewable[] objs, String[] name_objs, String[] functor_params, String code_body)
	{
		long[] p_objs = new long[objs.length];
		for (int i = 0; i<objs.length; i++ )
			p_objs[i] = objs[i].cptr();
		return Native.functor_create(p_objs, name_objs, functor_params, code_body);
	}

	public Functor(DeviceViewable[] objs, String[] name_objs, String[] functor_params, String code_body)
	{
		super(create(objs, name_objs, functor_params, code_body));
		m_objs = objs;
	}

	public Functor(String name_built_in_view_cls)
	{
		super(Native.built_in_functor_create(name_built_in_view_cls));
		m_objs = null;
	}

	private DeviceViewable[] m_objs;

	@Override
    public void close()
    {
    	m_objs = null;
        super.close();
    }
}
