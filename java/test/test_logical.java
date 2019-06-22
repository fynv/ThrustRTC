import java.util.*;
import JThrustRTC.*;

public class test_logical
{
	public static void main(String[] args) 
	{
        Functor identity= new Functor("Identity");
        DVVector d_A = new DVVector(new byte[] { 1, 1, 0 });

        System.out.println(TRTC.All_Of(d_A.range(0,2), identity));
        System.out.println(TRTC.All_Of(d_A.range(0,3), identity));
        System.out.println(TRTC.All_Of(d_A.range(0,0), identity));

        System.out.println(TRTC.Any_Of(d_A.range(0, 2), identity));
        System.out.println(TRTC.Any_Of(d_A.range(0, 3), identity));
        System.out.println(TRTC.Any_Of(d_A.range(2, 3), identity));
        System.out.println(TRTC.Any_Of(d_A.range(0, 0), identity));

        System.out.println(TRTC.None_Of(d_A.range(0, 2), identity));
        System.out.println(TRTC.None_Of(d_A.range(0, 3), identity));
        System.out.println(TRTC.None_Of(d_A.range(2, 3), identity));
        System.out.println(TRTC.None_Of(d_A.range(0, 0), identity));
	}
}
