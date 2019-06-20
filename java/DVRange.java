package JThrustRTC;

public class DVRange extends DVVectorLike
{

	public DVRange(DVVectorLike src, int begin, int end)
	{
		super(Native.dvrange_create(src.cptr(), begin, end));
	    m_vec_src = src;
	}

	public DVRange(DVVectorLike src)
	{
		super(Native.dvrange_create(src.cptr(), 0, -1));
	    m_vec_src = src;
	}

	private final DVVectorLike m_vec_src;
}
