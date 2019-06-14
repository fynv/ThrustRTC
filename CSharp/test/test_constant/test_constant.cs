using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_constant
{
    class test_constant
    {
        static void Main(string[] args)
        {
            DVConstant src = new DVConstant(new DVInt32(123), 10);
            DVVector dst = new DVVector("int32_t", 10);
            TRTC.Copy(src, dst);
            print_array((int[])dst.to_host());
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
