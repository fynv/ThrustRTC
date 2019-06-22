import java.util.*;
import JThrustRTC.*;

public class test_copy
{
	public static void main(String[] args) 
	{
		{
			DVVector dIn = new DVVector(new int[] { 10, 20, 30, 40, 50, 60, 70, 80 });
			DVVector dOut = new DVVector("int32_t", 8);
			TRTC.Copy(dIn, dOut);
			System.out.println(Arrays.toString((int[])dOut.to_host()));
		}

		Functor is_even = new Functor(new String[]{ "x" }, "        return x % 2 == 0;\n");

		{
			DVVector dIn = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
			DVVector dOut = new DVVector("int32_t", 6);
			int count = TRTC.Copy_If(dIn, dOut, is_even);
			System.out.println(Arrays.toString((int[])dOut.to_host(0, count)));
		}

		{
			DVVector dIn = new DVVector(new int[] { 0, 1, 2, 3, 4, 5 });
			DVVector dStencil = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
			DVVector dOut = new DVVector("int32_t", 6);
			int count = TRTC.Copy_If_Stencil(dIn, dStencil, dOut, is_even);
			System.out.println(Arrays.toString((int[])dOut.to_host(0, count)));
		}
	}
}
