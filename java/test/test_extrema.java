import java.util.*;
import JThrustRTC.*;

public class test_extrema
{
	public static void main(String[] args) 
	{
        DVVector d_data = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
        System.out.println(TRTC.Min_Element(d_data));
        System.out.println(TRTC.Max_Element(d_data));
        System.out.println(Arrays.toString(TRTC.MinMax_Element(d_data)));
	}
}
