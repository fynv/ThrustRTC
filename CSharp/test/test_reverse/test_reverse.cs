using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_reverse
{
    class test_reverse
    {
        static void Main(string[] args)
        {
            DVVector dvalues = new DVVector(new int[] { 3, 7, 2, 5 });
            DVReverse src = new DVReverse(dvalues);
            DVVector dst = new DVVector("int32_t", 4);
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
