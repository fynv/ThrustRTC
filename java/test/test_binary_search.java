import java.util.*;
import JThrustRTC.*;

public class test_binary_search
{
	public static void main(String[] args) 
	{
		DVVector d_input = new DVVector(new int[] { 0, 2, 5, 7, 8 });
		{
			System.out.println(TRTC.Lower_Bound(d_input, new DVInt32(0)));
			System.out.println(TRTC.Lower_Bound(d_input, new DVInt32(1)));
			System.out.println(TRTC.Lower_Bound(d_input, new DVInt32(2)));
			System.out.println(TRTC.Lower_Bound(d_input, new DVInt32(3)));
			System.out.println(TRTC.Lower_Bound(d_input, new DVInt32(8)));
			System.out.println(TRTC.Lower_Bound(d_input, new DVInt32(9)));
		}
		System.out.println("");
		{
			System.out.println(TRTC.Upper_Bound(d_input, new DVInt32(0)));
			System.out.println(TRTC.Upper_Bound(d_input, new DVInt32(1)));
			System.out.println(TRTC.Upper_Bound(d_input, new DVInt32(2)));
			System.out.println(TRTC.Upper_Bound(d_input, new DVInt32(3)));
			System.out.println(TRTC.Upper_Bound(d_input, new DVInt32(8)));
			System.out.println(TRTC.Upper_Bound(d_input, new DVInt32(9)));
		}
		System.out.println("");
		{
			System.out.println(TRTC.Binary_Search(d_input, new DVInt32(0)));
			System.out.println(TRTC.Binary_Search(d_input, new DVInt32(1)));
			System.out.println(TRTC.Binary_Search(d_input, new DVInt32(2)));
			System.out.println(TRTC.Binary_Search(d_input, new DVInt32(3)));
			System.out.println(TRTC.Binary_Search(d_input, new DVInt32(8)));
			System.out.println(TRTC.Binary_Search(d_input, new DVInt32(9)));
		}
		System.out.println("");

		DVVector d_values = new DVVector(new int[] { 0, 1, 2, 3, 8, 9 });
		DVVector d_output = new DVVector("int32_t", 6);

		TRTC.Lower_Bound_V(d_input, d_values, d_output);
		System.out.println(Arrays.toString((int[])d_output.to_host()));
		
		TRTC.Upper_Bound_V(d_input, d_values, d_output);
		System.out.println(Arrays.toString((int[])d_output.to_host()));

		TRTC.Binary_Search_V(d_input, d_values, d_output);
		System.out.println(Arrays.toString((int[])d_output.to_host()));
	}
}
