package JThrustRTC;

public class DVConstant extends DVVectorLike
{
	public DVConstant(DeviceViewable dvobj, int size)
	{
	    super(Native.dvconstant_create(dvobj.cptr(), size));
	    m_dvobj = dvobj;
	}

	public DVConstant(DeviceViewable dvobj)
	{
	    this(dvobj, -1);
	}

	private DeviceViewable m_dvobj;

	@Override
    public void close()
    {
    	m_dvobj = null;
        super.close();
    }
}
