import java.util.*;
import JThrustRTC.*;

public class test_sort
{
	public static void main(String[] args) 
	{
        {
            DVVector dvalues = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
            System.out.println(TRTC.Is_Sorted(dvalues));
            TRTC.Sort(dvalues);
            System.out.println(Arrays.toString((int[])dvalues.to_host()));
            System.out.println(TRTC.Is_Sorted(dvalues));
        }

        {
            Functor comp = new Functor("Greater");
            DVVector dvalues = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
            System.out.println(TRTC.Is_Sorted(dvalues, comp));
            TRTC.Sort(dvalues, comp);
           	System.out.println(Arrays.toString((int[])dvalues.to_host()));
            System.out.println(TRTC.Is_Sorted(dvalues, comp));
        }

        {
            DVVector dkeys = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
            DVVector dvalues = new DVVector(new int[] { 1, 2, 3, 4, 5, 6 });
            TRTC.Sort_By_Key(dkeys, dvalues);
            System.out.println(Arrays.toString((int[])dkeys.to_host()));
            System.out.println(Arrays.toString((int[])dvalues.to_host()));
        }

        {
            Functor comp = new Functor("Greater");
            DVVector dkeys = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
            DVVector dvalues = new DVVector(new int[] { 1, 2, 3, 4, 5, 6 });
            TRTC.Sort_By_Key(dkeys, dvalues, comp);
            System.out.println(Arrays.toString((int[])dkeys.to_host()));
            System.out.println(Arrays.toString((int[])dvalues.to_host()));
        }

        {
            DVVector dvalues = new DVVector(new int[] { 0, 1, 2, 3, 0, 1, 2, 3 });
            System.out.println(TRTC.Is_Sorted_Until(dvalues));
        }

        {
            DVVector dvalues = new DVVector(new int[] { 3, 2, 1, 0, 3, 2, 1, 0 });
            System.out.println(TRTC.Is_Sorted_Until(dvalues, new Functor("Greater")));
        }
	}
}
