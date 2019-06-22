import java.util.*;
import JThrustRTC.*;

public class test_scan
{
	public static void main(String[] args) 
	{
		{
			DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
			TRTC.Inclusive_Scan(darr, darr);
			System.out.println(Arrays.toString((int[])darr.to_host()));
		}

		{
			DVVector darr = new DVVector(new int[] { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 });
			TRTC.Inclusive_Scan(darr, darr, new Functor("Maximum"));
			System.out.println(Arrays.toString((int[])darr.to_host()));
		}

		{
			DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
			TRTC.Exclusive_Scan(darr, darr);
			System.out.println(Arrays.toString((int[])darr.to_host()));
		}

		{
			DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
			TRTC.Exclusive_Scan(darr, darr, new DVInt32(4));
			System.out.println(Arrays.toString((int[])darr.to_host()));
		}

		{
			DVVector darr = new DVVector(new int[] { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 });
			TRTC.Exclusive_Scan(darr, darr, new DVInt32(1), new Functor("Maximum"));
			System.out.println(Arrays.toString((int[])darr.to_host()));
		}

	}
}
