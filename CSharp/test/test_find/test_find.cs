using System;
using ThrustRTCSharp;

namespace test_find
{
    class test_find
    {
        static void Main(string[] args)
        {
            DVVector d_values = new DVVector(new int[] { 0, 5, 3, 7 });
            Console.WriteLine(TRTC.Find(d_values, new DVInt32(3)));
            Console.WriteLine(TRTC.Find(d_values, new DVInt32(5)));
            Console.WriteLine(TRTC.Find(d_values, new DVInt32(9)));
        }
    }
}
