package JThrustRTC;

public class DVInt64 extends DeviceViewable
{
	public DVInt64(long v)
	{
		super(Native.dvint64_create(v));
	}

	public long value()
	{
		return Native.dvint64_value(cptr());
	}
}
