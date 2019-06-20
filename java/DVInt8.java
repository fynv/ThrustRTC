package JThrustRTC;

public class DVInt8 extends DeviceViewable
{
	public DVInt8(byte v)
	{
		super(Native.dvint8_create(v));
	}

	public byte value()
	{
		return Native.dvint8_value(cptr());
	}
}
