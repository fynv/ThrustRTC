import java.util.*;
import JThrustRTC.*;

public class test_scan_by_key
{
	public static void main(String[] args) 
	{
		{
			DVVector d_keys = new DVVector(new int[] { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 });
			DVVector d_values = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
			TRTC.Inclusive_Scan_By_Key(d_keys, d_values, d_values);
			System.out.println(Arrays.toString((int[])d_values.to_host()));
		}

		{
			DVVector d_keys = new DVVector(new int[] { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 });
			DVVector d_values = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
			TRTC.Exclusive_Scan_By_Key(d_keys, d_values, d_values);
			System.out.println(Arrays.toString((int[])d_values.to_host()));
		}

		{
			DVVector d_keys = new DVVector(new int[] { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 });
			DVVector d_values = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
			TRTC.Exclusive_Scan_By_Key(d_keys, d_values, d_values, new DVInt32(5));
			System.out.println(Arrays.toString((int[])d_values.to_host()));
		}
	}
}
