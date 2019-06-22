import java.util.*;
import JThrustRTC.*;

public class test_tabulate
{
	public static void main(String[] args) 
	{	
        DVVector vec = new DVVector("int32_t", 10);
        TRTC.Sequence(vec);
        TRTC.Tabulate(vec, new Functor("Negate"));
        System.out.println(Arrays.toString((int[])vec.to_host()));
	}	

}
