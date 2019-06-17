using System;
using ThrustRTCSharp;

namespace test_merge
{
    class test_merge
    {
        static void Main(string[] args)
        {
            {
                DVVector dIn1 = new DVVector(new int[] { 1, 3, 5, 7, 9, 11 });
                DVVector dIn2 = new DVVector(new int[] { 1, 1, 2, 3, 5, 8, 13 });
                DVVector dOut = new DVVector("int32_t", 13);
                TRTC.Merge(dIn1, dIn2, dOut);
                print_array((int[])dOut.to_host());
            }

            {
                DVVector dIn1 = new DVVector(new int[] { 11, 9, 7, 5, 3, 1 });
                DVVector dIn2 = new DVVector(new int[] { 13, 8, 5, 3, 2, 1, 1 });
                DVVector dOut = new DVVector("int32_t", 13);
                TRTC.Merge(dIn1, dIn2, dOut, new Functor("Greater"));
                print_array((int[])dOut.to_host());
            }

            {
                DVVector dKeys1 = new DVVector(new int[] { 1, 3, 5, 7, 9, 11 });
                DVVector dVals1 = new DVVector(new int[] { 0, 0, 0, 0, 0, 0 });
                DVVector dKeys2 = new DVVector(new int[] { 1, 1, 2, 3, 5, 8, 13 });
                DVVector dVals2 = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1 });
                DVVector dKeysOut = new DVVector("int32_t", 13);
                DVVector dValsOut = new DVVector("int32_t", 13);
                TRTC.Merge_By_Key(dKeys1, dKeys2, dVals1, dVals2, dKeysOut, dValsOut);
                print_array((int[])dKeysOut.to_host());
                print_array((int[])dValsOut.to_host());
            }

            {
                DVVector dKeys1 = new DVVector(new int[] { 11, 9, 7, 5, 3, 1 });
                DVVector dVals1 = new DVVector(new int[] { 0, 0, 0, 0, 0, 0 });
                DVVector dKeys2 = new DVVector(new int[] { 13, 8, 5, 3, 2, 1, 1 });
                DVVector dVals2 = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1 });
                DVVector dKeysOut = new DVVector("int32_t", 13);
                DVVector dValsOut = new DVVector("int32_t", 13);
                TRTC.Merge_By_Key(dKeys1, dKeys2, dVals1, dVals2, dKeysOut, dValsOut, new Functor("Greater"));
                print_array((int[])dKeysOut.to_host());
                print_array((int[])dValsOut.to_host());
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
