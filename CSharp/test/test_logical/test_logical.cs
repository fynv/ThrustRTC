using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_logical
{
    class test_logical
    {
        static void Main(string[] args)
        {
            Functor identity= new Functor("Identity");
            DVVector d_A = new DVVector(new bool[] { true, true, false });

            Console.WriteLine(TRTC.All_Of(d_A, identity, 0, 2));
            Console.WriteLine(TRTC.All_Of(d_A, identity, 0, 3));
            Console.WriteLine(TRTC.All_Of(d_A, identity, 0, 0));

            Console.WriteLine(TRTC.Any_Of(d_A, identity, 0, 2));
            Console.WriteLine(TRTC.Any_Of(d_A, identity, 0, 3));
            Console.WriteLine(TRTC.Any_Of(d_A, identity, 2, 3));
            Console.WriteLine(TRTC.Any_Of(d_A, identity, 0, 0));

            Console.WriteLine(TRTC.None_Of(d_A, identity, 0, 2));
            Console.WriteLine(TRTC.None_Of(d_A, identity, 0, 3));
            Console.WriteLine(TRTC.None_Of(d_A, identity, 2, 3));
            Console.WriteLine(TRTC.None_Of(d_A, identity, 0, 0));
        }
    }
}
