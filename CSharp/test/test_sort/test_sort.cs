using System;
using ThrustRTCSharp;

namespace test_sort
{
    class test_sort
    {
        static void Main(string[] args)
        {
            {
                DVVector dvalues = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
                Console.WriteLine(TRTC.Is_Sorted(dvalues));
                TRTC.Sort(dvalues);
                print_array((int[])dvalues.to_host());
                Console.WriteLine(TRTC.Is_Sorted(dvalues));
            }

            {
                Functor comp = new Functor("Greater");
                DVVector dvalues = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
                Console.WriteLine(TRTC.Is_Sorted(dvalues, comp));
                TRTC.Sort(dvalues, comp);
                print_array((int[])dvalues.to_host());
                Console.WriteLine(TRTC.Is_Sorted(dvalues, comp));
            }

            {
                DVVector dkeys = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
                DVVector dvalues = new DVVector(new int[] { 1, 2, 3, 4, 5, 6 });
                TRTC.Sort_By_Key(dkeys, dvalues);
                print_array((int[])dkeys.to_host());
                print_array((int[])dvalues.to_host());
            }

            {
                Functor comp = new Functor("Greater");
                DVVector dkeys = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
                DVVector dvalues = new DVVector(new int[] { 1, 2, 3, 4, 5, 6 });
                TRTC.Sort_By_Key(dkeys, dvalues, comp);
                print_array((int[])dkeys.to_host());
                print_array((int[])dvalues.to_host());
            }

            {
                DVVector dvalues = new DVVector(new int[] { 0, 1, 2, 3, 0, 1, 2, 3 });
                Console.WriteLine(TRTC.Is_Sorted_Until(dvalues));
            }

            {
                DVVector dvalues = new DVVector(new int[] { 3, 2, 1, 0, 3, 2, 1, 0 });
                Console.WriteLine(TRTC.Is_Sorted_Until(dvalues, new Functor("Greater")));
            }
        }

        static void print_array<T>(T[] arr)
        {
            foreach (var item in arr)
            {
                Console.Write(item.ToString());
                Console.Write(" ");
            }
            Console.WriteLine("");
        }
    }
}
