import java.util.*;
import JThrustRTC.*;

public class test_reduce
{
	public static void main(String[] args) 
	{
        {
            DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
           	System.out.println(TRTC.Reduce(darr));
            System.out.println(TRTC.Reduce(darr, new DVInt32(1)));
           	System.out.println(TRTC.Reduce(darr, new DVInt32(-1), new Functor("Maximum")));
        }

        {
            DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
            DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
            DVVector d_keys_out = new DVVector("int32_t", 7);
            DVVector d_values_out = new DVVector("int32_t", 7);
            int count = TRTC.Reduce_By_Key(d_keys_in, d_values_in, d_keys_out, d_values_out);
            System.out.println(Arrays.toString((int[])d_keys_out.to_host(0, count)));
            System.out.println(Arrays.toString((int[])d_values_out.to_host(0, count)));
        }

        {
            DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
            DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
            DVVector d_keys_out = new DVVector("int32_t", 7);
            DVVector d_values_out = new DVVector("int32_t", 7);
            int count = TRTC.Reduce_By_Key(d_keys_in, d_values_in, d_keys_out, d_values_out, new Functor("EqualTo"));
            System.out.println(Arrays.toString((int[])d_keys_out.to_host(0, count)));
            System.out.println(Arrays.toString((int[])d_values_out.to_host(0, count)));
        }

        {
            DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
            DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
            DVVector d_keys_out = new DVVector("int32_t", 7);
            DVVector d_values_out = new DVVector("int32_t", 7);
            int count = TRTC.Reduce_By_Key(d_keys_in, d_values_in, d_keys_out, d_values_out, new Functor("EqualTo"), new Functor("Plus"));
            System.out.println(Arrays.toString((int[])d_keys_out.to_host(0, count)));
            System.out.println(Arrays.toString((int[])d_values_out.to_host(0, count)));
        }
	}
}
