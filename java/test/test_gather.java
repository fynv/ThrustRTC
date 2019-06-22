import java.util.*;
import JThrustRTC.*;

public class test_gather
{
	public static void main(String[] args) 
	{	
        {
            DVVector d_values = new DVVector(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
            DVVector d_map = new DVVector(new int[] { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 });
            DVVector d_output = new DVVector("int32_t", 10);
            TRTC.Gather(d_map, d_values, d_output);
            System.out.println(Arrays.toString((int[])d_output.to_host()));
        }

        {
            DVVector d_values = new DVVector(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            DVVector d_stencil = new DVVector(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
            DVVector d_map = new DVVector(new int[] { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 });
            DVVector d_output = new DVVector("int32_t", 10);
            TRTC.Fill(d_output, new DVInt32(7));
            TRTC.Gather_If(d_map, d_stencil, d_values, d_output);
            System.out.println(Arrays.toString((int[])d_output.to_host()));
        }

        Functor is_even = new Functor( new String[] { "x" }, "        return ((x % 2) == 0);\n" );
        {
            DVVector d_values = new DVVector(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            DVVector d_stencil = new DVVector(new int[] { 0, 3, 4, 1, 4, 1, 2, 7, 8, 9 });
            DVVector d_map = new DVVector(new int[] { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 });
            DVVector d_output = new DVVector("int32_t", 10);
            TRTC.Fill(d_output, new DVInt32(7));
            TRTC.Gather_If(d_map, d_stencil, d_values, d_output, is_even);
            System.out.println(Arrays.toString((int[])d_output.to_host()));
        }
	}	

}
