import java.util.*;
import JThrustRTC.*;

public class test_find
{
	public static void main(String[] args) 
	{
		DVVector d_values = new DVVector(new int[] { 0, 5, 3, 7 });
		System.out.println(TRTC.Find(d_values, new DVInt32(3)));
		System.out.println(TRTC.Find(d_values, new DVInt32(5)));
		System.out.println(TRTC.Find(d_values, new DVInt32(9)));
	}
}