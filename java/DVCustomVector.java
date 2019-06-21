package JThrustRTC;

public class DVCustomVector extends DVVectorLike
{
    private static long create(DeviceViewable[] objs, String[] name_objs, String name_idx, String code_body, String elem_cls, int size, boolean read_only)
    {
    	long[] p_objs = new long[objs.length];
		for (int i = 0; i<objs.length; i++ )
			p_objs[i] = objs[i].cptr();
		return Native.dvcustomvector_create(p_objs, name_objs, name_idx, code_body, elem_cls, size, read_only);
    }

    public DVCustomVector(DeviceViewable[] objs, String[] name_objs, String name_idx, String code_body, String elem_cls, int size, boolean read_only)
    {
    	super(create(objs, name_objs, name_idx, code_body, elem_cls, size, read_only));
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
