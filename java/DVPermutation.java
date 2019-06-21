package JThrustRTC;

public class DVPermutation extends DVVectorLike
{
	public DVPermutation(DVVectorLike vec_value, DVVectorLike vec_index)
	{
	    super(Native.dvpermutation_create(vec_value.cptr(), vec_index.cptr()));
	    m_vec_value = vec_value;
	    m_vec_index = vec_index;
	}

	private DVVectorLike m_vec_value;
    private DVVectorLike m_vec_index;

	@Override
    public void close()
    {
  		m_vec_value = null;
        m_vec_index = null;
        super.close();
    }

}
