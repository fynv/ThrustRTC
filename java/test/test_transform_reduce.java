import java.util.*;
import JThrustRTC.*;

public class test_transform_reduce
{
	public static void main(String[] args) 
	{
        Functor absolute_value = new Functor ( new String[]{ "x" }, "        return x<(decltype(x))0 ? -x : x;\n" );
        DVVector d_data = new DVVector(new int[] { -1, 0, -2, -2, 1, -3 });
        System.out.println(TRTC.Transform_Reduce(d_data, absolute_value, new DVInt32(0), new Functor("Maximum")));
	}
}
