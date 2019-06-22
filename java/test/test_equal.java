import java.util.*;
import JThrustRTC.*;

public class test_equal
{
	public static void main(String[] args) 
	{
        {
            DVVector darr1 = new DVVector(new int[] { 3, 1, 4, 1, 5, 9, 3 });
            DVVector darr2 = new DVVector(new int[] { 3, 1, 4, 2, 8, 5, 7 });
            DVVector darr3 = new DVVector(new int[] { 3, 1, 4, 1, 5, 9, 3 });
            System.out.println(TRTC.Equal(darr1, darr2).toString() + " " +TRTC.Equal(darr1, darr3).toString());
        }

        {
            Functor compare_modulo_two = new Functor( new String[]{ "x", "y" }, "        return (x % 2) == (y % 2);\n" );
            DVVector dx = new DVVector(new int[] { 1, 2, 3, 4, 5, 6 });
            DVVector dy = new DVVector(new int[] { 7, 8, 9, 10, 11, 12 });
            System.out.println(TRTC.Equal(dx, dy, compare_modulo_two));
        }
	}
}
