import java.util.*;
import JThrustRTC.*;

public class test_transform
{
	public static void main(String[] args) 
	{	
        Functor is_odd = new Functor(new String[] { "x" }, "        return x % 2;\n");

        {
            DVVector vec = new DVVector(new int[] { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 });
            TRTC.Transform(vec, vec, new Functor("Negate")); // in place
        	System.out.println(Arrays.toString((int[])vec.to_host()));
        }

        {
            DVVector d_in1 = new DVVector(new int[] { -5, 0, 2, 3, 2, 4 });
            DVVector d_in2 = new DVVector(new int[] { 3, 6, -2, 1, 2, 3 });
            DVVector d_out = new DVVector("int32_t", 6);
            TRTC.Transform_Binary(d_in1, d_in2, d_out, new Functor("Plus"));
            System.out.println(Arrays.toString((int[])d_out.to_host()));
        }

        {
            DVVector vec = new DVVector(new int[] { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 });
            TRTC.Transform_If(vec, vec, new Functor("Negate"), is_odd); // in place
            System.out.println(Arrays.toString((int[])vec.to_host()));
        }

        {
            DVVector d_data = new DVVector(new int[] { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 });
            DVVector d_stencil = new DVVector(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
            TRTC.Transform_If_Stencil(d_data, d_stencil, d_data, new Functor("Negate"), new Functor("Identity")); // in place
            System.out.println(Arrays.toString((int[])d_data.to_host()));
        }

        {
            DVVector d_in1 = new DVVector(new int[] { -5, 0, 2, 3, 2, 4 });
            DVVector d_in2 = new DVVector(new int[] { 3, 6, -2, 1, 2, 3 });
            DVVector d_stencil = new DVVector(new int[] { 1, 0, 1, 0, 1, 0 });
            DVVector d_output = new DVVector("int32_t", 6);
            TRTC.Transform_Binary_If_Stencil(d_in1, d_in2, d_stencil, d_output, new Functor("Plus"), new Functor("Identity")); // in place
            System.out.println(Arrays.toString((int[])d_output.to_host()));
        }

	}	

}
