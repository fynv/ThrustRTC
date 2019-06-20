package JThrustRTC;

public class DVVector extends DVVectorLike
{
	public DVVector(String elem_cls, int size)
	{
		super( Native.dvvector_create(elem_cls, size) );
	}

	public DVVector(byte[] hdata)
	{
		super( Native.dvvector_create(hdata) );
	}

	public DVVector(short[] hdata)
	{
		super( Native.dvvector_create(hdata) );
	}

	public DVVector(int[] hdata)
	{
		super( Native.dvvector_create(hdata) );
	}

	public DVVector(long[] hdata)
	{
		super( Native.dvvector_create(hdata) );
	}

	public DVVector(float[] hdata)
	{
		super( Native.dvvector_create(hdata) );
	}

	public DVVector(double[] hdata)
	{
		super( Native.dvvector_create(hdata) );
	}

	public Object to_host(int begin, int end)
	{
		String type = name_elem_cls();
		if (end == -1) end = size();
        if (end < begin) return null;
        int h_size = end - begin;

        if (type.equals("int8_t"))
        {
        	byte[] arr = new byte[h_size];
        	if (h_size>0)
            {
            	Native.dvvector_to_host(cptr(), arr, begin, end);
                return arr;
            }
        }
        else if (type.equals("int16_t"))
        {
        	short[] arr = new short[h_size];
        	if (h_size>0)
            {
            	Native.dvvector_to_host(cptr(), arr, begin, end);
                return arr;
            }
        }
        else if (type.equals("int32_t"))
        {
        	int[] arr = new int[h_size];
        	if (h_size>0)
            {
            	Native.dvvector_to_host(cptr(), arr, begin, end);
                return arr;
            }
        }
        else if (type.equals("int64_t"))
        {
        	long[] arr = new long[h_size];
        	if (h_size>0)
            {
            	Native.dvvector_to_host(cptr(), arr, begin, end);
                return arr;
            }
        }
        else if (type.equals("float"))
        {
        	float[] arr = new float[h_size];
        	if (h_size>0)
            {
            	Native.dvvector_to_host(cptr(), arr, begin, end);
                return arr;
            }
        }
        else if (type.equals("double"))
        {
        	double[] arr = new double[h_size];
        	if (h_size>0)
            {
            	Native.dvvector_to_host(cptr(), arr, begin, end);
                return arr;
            }
        }

        return null;
	}

	public Object to_host()
	{
		return to_host(0, -1);
	}


}