package JThrustRTC;

public class DVReverse extends DVVectorLike
{
	public DVReverse(DVVectorLike vec_value)
	{
		super(Native.dvreverse_create(vec_value.cptr()));
	    m_vec_value = vec_value;
	}

	private DVVectorLike m_vec_value;

	@Override
    public void close()
    {
  		m_vec_value = null;
        super.close();
    }
}
