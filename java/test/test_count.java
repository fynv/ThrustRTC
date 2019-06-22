import java.util.*;
import JThrustRTC.*;

public class test_count
{
	public static void main(String[] args) 
	{
        int[] hin=new int[2000];
        for (int i = 0; i < 2000; i++)
            hin[i] = i % 100;

        DVVector din = new DVVector(hin);
		System.out.println(TRTC.Count(din, new DVInt32(47)));

        TRTC.Sequence(din);
        Functor op = new Functor(new String[]{ "x" }, "        return (x%100)==47;\n" );
		System.out.println(TRTC.Count_If(din, op));
	}
}
