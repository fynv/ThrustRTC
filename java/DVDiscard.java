package JThrustRTC;

public class DVDiscard extends DVVectorLike
{
	public DVDiscard(String elem_cls, int size)
	{
		super(Native.dvdiscard_create(elem_cls, size));			
	}

	public DVDiscard(String elem_cls)
	{
		this(elem_cls, -1);
	}
}
