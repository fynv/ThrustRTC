using System;
using ThrustRTCSharp;

namespace test_logical
{
    class test_logical
    {
        static void Main(string[] args)
        {
            Functor identity= new Functor("Identity");
            DVVector d_A = new DVVector(new bool[] { true, true, false });

            Console.WriteLine(TRTC.All_Of(d_A.range(0,2), identity));
            Console.WriteLine(TRTC.All_Of(d_A.range(0,3), identity));
            Console.WriteLine(TRTC.All_Of(d_A.range(0,0), identity));

            Console.WriteLine(TRTC.Any_Of(d_A.range(0, 2), identity));
            Console.WriteLine(TRTC.Any_Of(d_A.range(0, 3), identity));
            Console.WriteLine(TRTC.Any_Of(d_A.range(2, 3), identity));
            Console.WriteLine(TRTC.Any_Of(d_A.range(0, 0), identity));

            Console.WriteLine(TRTC.None_Of(d_A.range(0, 2), identity));
            Console.WriteLine(TRTC.None_Of(d_A.range(0, 3), identity));
            Console.WriteLine(TRTC.None_Of(d_A.range(2, 3), identity));
            Console.WriteLine(TRTC.None_Of(d_A.range(0, 0), identity));
        }
    }
}
