package JThrustRTC;

public class DVDouble extends DeviceViewable
{
	public DVDouble(double v)
	{
		super(Native.dvdouble_create(v));
	}

	public double value()
	{
		return Native.dvdouble_value(cptr());
	}
}
