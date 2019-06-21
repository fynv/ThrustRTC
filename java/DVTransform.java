package JThrustRTC;

public class DVTransform extends DVVectorLike
{
	public DVTransform(DVVectorLike vec_in, String elem_cls, Functor op)
	{
		super(Native.dvtransform_create(vec_in.cptr(), elem_cls, op.cptr()));
		m_vec_in = vec_in;
		m_op = op;
	}

	private DVVectorLike m_vec_in;
	private Functor m_op;

	@Override
    public void close()
    {
        m_vec_in = null;
        m_op = null;
        super.close();
    }
}
