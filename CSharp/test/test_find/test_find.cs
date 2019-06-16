using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
