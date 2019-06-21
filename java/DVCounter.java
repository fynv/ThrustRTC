package JThrustRTC;

public class DVCounter extends DVVectorLike
{
	public DVCounter(DeviceViewable dvobj_init, int size)
	{
	    super(Native.dvcounter_create(dvobj_init.cptr(), size));
	    m_dvobj_init = dvobj_init;
	}

	public DVCounter(DeviceViewable dvobj)
	{
	    this(dvobj, -1);
	}

	private DeviceViewable m_dvobj_init;

	@Override
    public void close()
    {
    	m_dvobj_init = null;
        super.close();
    }

}