import java.util.*;
import JThrustRTC.*;

public class test_discard
{
	public static void main(String[] args) 
	{
		TRTC.Set_Verbose();

		// just to verify that it compiles
		DVCounter src = new DVCounter(new DVInt32(5), 10);
		DVDiscard sink = new DVDiscard("int32_t");
		TRTC.Copy(src, sink);
	}

}
