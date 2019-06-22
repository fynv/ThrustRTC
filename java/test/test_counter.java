import java.util.*;
import JThrustRTC.*;

public class test_counter
{
	public static void main(String[] args) 
	{
		DVCounter src = new DVCounter(new DVInt32(1), 10);
		DVVector dst = new DVVector("int32_t", 10);
		TRTC.Copy(src, dst);
		System.out.println(Arrays.toString((int[])dst.to_host()));
	}

}
