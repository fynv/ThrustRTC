package JThrustRTC;

public class DVFloat extends DeviceViewable
{
	public DVFloat(float v)
	{
		super(Native.dvfloat_create(v));
	}

	public float value()
	{
		return Native.dvfloat_value(cptr());
	}
}
