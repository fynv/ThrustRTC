import java.util.*;
import JThrustRTC.*;

public class test_constant
{
	public static void main(String[] args) 
	{
		DVConstant src = new DVConstant(new DVInt32(123), 10);
		DVVector dst = new DVVector("int32_t", 10);
		TRTC.Copy(src, dst);
		System.out.println(Arrays.toString((int[])dst.to_host()));
	}

}
