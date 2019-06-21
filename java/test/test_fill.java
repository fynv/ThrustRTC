import java.util.*;
import JThrustRTC.*;

public class test_fill
{
	public static void main(String[] args) 
	{
		DVVector vec_to_fill = new DVVector("int32_t", 5);
		TRTC.Fill(vec_to_fill, new DVInt32(123));
		System.out.println(Arrays.toString((int[])vec_to_fill.to_host()));
	}	

}
