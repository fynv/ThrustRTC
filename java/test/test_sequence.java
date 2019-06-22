import java.util.*;
import JThrustRTC.*;

public class test_sequence
{
	public static void main(String[] args) 
	{
		  DVVector vec= new DVVector("int32_t", 10);

		  TRTC.Sequence(vec);
		  System.out.println(Arrays.toString((int[])vec.to_host()));

		  TRTC.Sequence(vec, new DVInt32(1));
		  System.out.println(Arrays.toString((int[])vec.to_host()));
		  
		  TRTC.Sequence(vec, new DVInt32(1), new DVInt32(3));
		  System.out.println(Arrays.toString((int[])vec.to_host()));

	}	

}
