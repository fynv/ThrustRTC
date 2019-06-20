package JThrustRTC;

public class DVInt32 extends DeviceViewable
{
	public DVInt32(int v)
	{
		super(Native.dvint32_create(v));
	}

	public int value()
	{
		return Native.dvint32_value(cptr());
	}
}
