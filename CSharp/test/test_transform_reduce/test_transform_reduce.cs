using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_transform_reduce
{
    class test_transform_reduce
    {
        static void Main(string[] args)
        {
            Functor absolute_value = new Functor ( new string[]{ "x" }, "        return x<(decltype(x))0 ? -x : x;\n" );
            DVVector d_data = new DVVector(new int[] { -1, 0, -2, -2, 1, -3 });
            Console.WriteLine(TRTC.Transform_Reduce(d_data, absolute_value, new DVInt32(0), new Functor("Maximum")));
        }
    }
}
