import java.util.*;
import JThrustRTC.*;

public class test_replace
{
	public static void main(String[] args) 
	{
		Functor is_less_than_zero = new Functor(new String[]{ "x" }, "        return x<0;\n" );

		{
			DVVector vec = new DVVector(new int[] { 1, 2, 3, 1, 2 });
			TRTC.Replace(vec, new DVInt32(1), new DVInt32(99));
			System.out.println(Arrays.toString((int[])vec.to_host()));
		}

		{
			DVVector vec = new DVVector(new int[] { 1, -2, 3, -4, 5 });
			TRTC.Replace_If(vec, is_less_than_zero, new DVInt32(99));
			System.out.println(Arrays.toString((int[])vec.to_host()));
		}

		{
			DVVector vec_in = new DVVector(new int[] { 1, 2, 3, 1, 2 });
			DVVector vec_out = new DVVector("int32_t", 5);
			TRTC.Replace_Copy(vec_in, vec_out, new DVInt32(1), new DVInt32(99));
			System.out.println(Arrays.toString((int[])vec_out.to_host()));
		}

		{
			DVVector vec_in = new DVVector(new int[] { 1, -2, 3, -4, 5 });
			DVVector vec_out = new DVVector("int32_t", 5);
			TRTC.Replace_Copy_If(vec_in, vec_out, is_less_than_zero, new DVInt32(99));
			System.out.println(Arrays.toString((int[])vec_out.to_host()));
		}
	}	

}
