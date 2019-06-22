import java.util.*;
import JThrustRTC.*;

public class test_mismatch
{
	public static void main(String[] args) 
	{
		{
			DVVector d1 = new DVVector(new int[] { 0, 5, 3, 7 });
			DVVector d2 = new DVVector(new int[] { 0, 5, 8, 7 });
			System.out.println(TRTC.Mismatch(d1, d2));
		}

		{
			DVVector d1 = new DVVector(new int[] { 0, 5, 3, 7 });
			DVVector d2 = new DVVector(new int[] { 0, 5, 8, 7 });
			System.out.println(TRTC.Mismatch(d1, d2, new Functor("EqualTo")));
		}
	}
}
