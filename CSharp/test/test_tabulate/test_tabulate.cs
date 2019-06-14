using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_tabulate
{
    class test_tabulate
    {
        static void Main(string[] args)
        {
            DVVector vec = new DVVector("int32_t", 10);
            TRTC.Sequence(vec);
            TRTC.Tabulate(vec, new Functor("Negate"));
            print_array((int[])vec.to_host());
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
