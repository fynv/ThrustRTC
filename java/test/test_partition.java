import java.util.*;
import JThrustRTC.*;

public class test_partition
{
	public static void main(String[] args) 
	{
		Functor is_even = new Functor( new String[]{ "x" }, "        return x % 2 == 0;\n" );

		{
			DVVector d_value = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
			TRTC.Partition(d_value, is_even);
			System.out.println(Arrays.toString((int[])d_value.to_host()));
		}

		{
			DVVector d_value = new DVVector(new int[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
			DVVector d_stencil = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
			TRTC.Partition_Stencil(d_value, d_stencil, is_even);
			System.out.println(Arrays.toString((int[])d_value.to_host()));
		}

		{
			DVVector d_value = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
			DVVector d_evens = new DVVector("int32_t", 10);
			DVVector d_odds = new DVVector("int32_t", 10);
			int count = TRTC.Partition_Copy(d_value, d_evens, d_odds, is_even);
			System.out.println(Arrays.toString((int[])d_evens.to_host(0, count)));
			System.out.println(Arrays.toString((int[])d_odds.to_host(0, 10 - count)));
		}

		{
			DVVector d_value = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
			DVVector d_stencil = new DVVector(new int[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
			DVVector d_evens = new DVVector("int32_t", 10);
			DVVector d_odds = new DVVector("int32_t", 10);
			int count = TRTC.Partition_Copy_Stencil(d_value, d_stencil, d_evens, d_odds, new Functor("Identity"));
			System.out.println(Arrays.toString((int[])d_evens.to_host(0, count)));
			System.out.println(Arrays.toString((int[])d_odds.to_host(0, 10 - count)));
		}

		{
			DVVector d_value = new DVVector(new int[] { 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 });
			System.out.println(TRTC.Partition_Point(d_value, is_even));
		}

		{
			DVVector d_value = new DVVector(new int[] { 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 });
			System.out.println(TRTC.Is_Partitioned(d_value, is_even));
		}

		{
			DVVector d_value = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
			System.out.println(TRTC.Is_Partitioned(d_value, is_even));
		}
	}

}