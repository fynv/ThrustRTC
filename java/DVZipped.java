package JThrustRTC;

public class DVZipped extends DVVectorLike
{
	private static long create(DVVectorLike[] vecs, String[] elem_names)
	{
		long[] p_vecs = new long[vecs.length];
        for (int i = 0; i < vecs.length; i++)
            p_vecs[i] = vecs[i].cptr();
        return Native.dvzipped_create(p_vecs, elem_names);
	}

	public DVZipped(DVVectorLike[] vecs, String[] elem_names)
	{
		super(create(vecs, elem_names));
		m_vecs = vecs;
	}

	private DVVectorLike[] m_vecs;

	@Override
    public void close()
    {
		m_vecs = null;
        super.close();
    }

}
