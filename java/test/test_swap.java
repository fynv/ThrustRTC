import java.util.*;
import JThrustRTC.*;

public class test_swap
{
	public static void main(String[] args) 
	{
		DVVector darr1 = new DVVector(new int[] { 10, 20, 30, 40, 50, 60, 70, 80 });
		DVVector darr2 = new DVVector(new int[] { 1000, 900, 800, 700, 600, 500, 400, 300 });
		TRTC.Swap(darr1, darr2);
		System.out.println(Arrays.toString((int[])darr1.to_host()));		
		System.out.println(Arrays.toString((int[])darr2.to_host()));		
	}
}
