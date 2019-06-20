package JThrustRTC;

public class DVInt16 extends DeviceViewable
{
	public DVInt16(short v)
	{
		super(Native.dvint16_create(v));
	}

	public short value()
	{
		return Native.dvint16_value(cptr());
	}
}
