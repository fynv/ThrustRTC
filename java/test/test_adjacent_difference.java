import java.util.*;
import JThrustRTC.*;

public class test_adjacent_difference
{
	public static void main(String[] args) 
	{
        {
            DVVector vec_in = new DVVector(new int[] { 1, 2, 1, 2, 1, 2, 1, 2 });
            DVVector vec_out = new DVVector("int32_t", 8);
            TRTC.Adjacent_Difference(vec_in, vec_out);
            System.out.println(Arrays.toString((int[])vec_out.to_host()));
        }

        {
            DVVector vec_in = new DVVector(new int[] { 1, 2, 1, 2, 1, 2, 1, 2 });
            DVVector vec_out = new DVVector("int32_t", 8);
            TRTC.Adjacent_Difference(vec_in, vec_out, new Functor("Plus"));
            System.out.println(Arrays.toString((int[])vec_out.to_host()));
        }
	}	

}
