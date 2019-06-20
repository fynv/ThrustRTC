package JThrustRTC;

public class DVVectorLike extends DeviceViewable
{
	public DVVectorLike(long _cptr)
	{
		super(_cptr);
	}

	public String name_elem_cls()
	{
		return Native.dvvectorlike_name_elem_cls(cptr());
	}

	public int size()
    {
        return Native.dvvectorlike_size(cptr());
    }

	public DVRange range(int begin, int end)
	{
		return new DVRange(this, begin, end);
	}
}

