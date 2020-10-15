package JThrustRTC;

public class DVVectorAdaptor extends DVVectorLike
{
	public DVVectorAdaptor(String elem_cls, int size, long native_pointer)
	{
		super( Native.dvvectoradaptor_create(elem_cls, size, native_pointer) );
	}

	public DVVectorAdaptor(DVVector vec, int begin, int end)
	{
		super( Native.dvvectoradaptor_create_from_dvvector(vec.cptr(), begin, end) );
	}

	public DVVectorAdaptor(DVVector vec)
	{
		super( Native.dvvectoradaptor_create_from_dvvector(vec.cptr(), 0, -1) );
	}

	public DVVectorAdaptor(DVVectorAdaptor vec, int begin, int end)
	{
		super( Native.dvvectoradaptor_create_from_dvvectoradaptor(vec.cptr(), begin, end) );
	}

	public DVVectorAdaptor(DVVectorAdaptor vec)
	{
		super( Native.dvvectoradaptor_create_from_dvvectoradaptor(vec.cptr(), 0, -1) );
	}

	public long native_pointer()
    {
    	return Native.dvvectoradaptor_native_pointer(cptr());
    }

}

