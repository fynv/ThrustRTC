import java.util.*;
import JThrustRTC.*;

public class test_permutation
{
	public static void main(String[] args) 
	{
		DVVector dvalues = new DVVector(new float[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f });
		DVVector dindices = new DVVector(new int[] { 2, 6, 1, 3 });
		DVPermutation src = new DVPermutation(dvalues, dindices);
		DVVector dst = new DVVector("float", 4);
		TRTC.Copy(src, dst);
		System.out.println(Arrays.toString((float[])dst.to_host()));
	}

}
