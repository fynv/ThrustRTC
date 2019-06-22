import java.util.*;
import JThrustRTC.*;

public class test_reverse
{
	public static void main(String[] args) 
	{
		DVVector dvalues = new DVVector(new int[] { 3, 7, 2, 5 });
		DVReverse src = new DVReverse(dvalues);
		DVVector dst = new DVVector("int32_t", 4);
		TRTC.Copy(src, dst);
		System.out.println(Arrays.toString((int[])dst.to_host()));
	}

}
