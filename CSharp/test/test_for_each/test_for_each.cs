using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_for_each
{
    class test_for_each
    {
        static void Main(string[] args)
        {
            Functor printf_functor = new Functor(new string[]{ "x" }, "        printf(\"%d\\n\", x);\n");
            DVVector vec = new DVVector(new int[] { 1, 2, 3, 1, 2 });
            TRTC.For_Each(vec, printf_functor);
        }
    }
}
