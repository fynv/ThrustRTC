import java.util.*;
import JThrustRTC.*;

public class test_remove
{
	public static void main(String[] args) 
	{
		{
			DVVector d_value = new DVVector(new int[] { 3, 1, 4, 1, 5, 9 });
			int count = TRTC.Remove(d_value, new DVInt32(1));
			System.out.println(Arrays.toString((int[])d_value.to_host(0, count)));
		}

		{
			DVVector d_in = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
			DVVector d_out = new DVVector("int32_t", 6);
			int count = TRTC.Remove_Copy(d_in, d_out, new DVInt32(0));
			System.out.println(Arrays.toString((int[])d_out.to_host(0, count)));
		}

		Functor is_even = new Functor(new String[] { "x" }, "        return x % 2 == 0;\n");

		{
			DVVector d_value = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
			int count = TRTC.Remove_If(d_value, is_even);
			System.out.println(Arrays.toString((int[])d_value.to_host(0, count)));
		}

		{
			DVVector d_in = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
			DVVector d_out = new DVVector("int32_t", 6);
			int count = TRTC.Remove_Copy_If(d_in, d_out, is_even);
			System.out.println(Arrays.toString((int[])d_out.to_host(0, count)));
		}

		Functor identity = new Functor("Identity");

		{
			DVVector d_value = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
			DVVector d_stencil = new DVVector(new int[] { 0, 1, 1, 1, 0, 0 });
			int count = TRTC.Remove_If_Stencil(d_value, d_stencil, identity);
			System.out.println(Arrays.toString((int[])d_value.to_host(0, count)));
		}

		{
			DVVector d_in = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
			DVVector d_stencil = new DVVector(new int[] { 1, 1, 0, 1, 0, 1 });
			DVVector d_out = new DVVector("int32_t", 6);
			int count = TRTC.Remove_Copy_If_Stencil(d_in, d_stencil, d_out, identity);
			System.out.println(Arrays.toString((int[])d_out.to_host(0, count)));
		}
	}

}