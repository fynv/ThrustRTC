using System;
using ThrustRTCSharp;

namespace test_mismatch
{
    class test_mismatch
    {
        static void Main(string[] args)
        {
            {
                DVVector d1 = new DVVector(new int[] { 0, 5, 3, 7 });
                DVVector d2 = new DVVector(new int[] { 0, 5, 8, 7 });
                Console.WriteLine(TRTC.Mismatch(d1, d2));
            }

            {
                DVVector d1 = new DVVector(new int[] { 0, 5, 3, 7 });
                DVVector d2 = new DVVector(new int[] { 0, 5, 8, 7 });
                Console.WriteLine(TRTC.Mismatch(d1, d2, new Functor("EqualTo")));
            }
        }
    }
}
