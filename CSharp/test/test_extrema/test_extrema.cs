using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_extrema
{
    class test_extrema
    {
        static void Main(string[] args)
        {
            DVVector d_data = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
            Console.WriteLine(TRTC.Min_Element(d_data));
            Console.WriteLine(TRTC.Max_Element(d_data));
            Console.WriteLine(TRTC.MinMax_Element(d_data));
        }
    }
}
