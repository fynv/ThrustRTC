using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_copy
{
    class test_copy
    {
        static void Main(string[] args)
        {
            {
                DVVector dIn = new DVVector(new int[] { 10, 20, 30, 40, 50, 60, 70, 80 });
                DVVector dOut = new DVVector("int32_t", 8);
                TRTC.Copy(dIn, dOut);
                print_array((int[])dOut.to_host());
            }

            Functor is_even = new Functor(new string[]{ "x" }, "        return x % 2 == 0;\n");

            {
                DVVector dIn = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
                DVVector dOut = new DVVector("int32_t", 6);
                long count = TRTC.Copy_If(dIn, dOut, is_even);
                print_array((int[])dOut.to_host(0, count));
            }

            {
                DVVector dIn = new DVVector(new int[] { 0, 1, 2, 3, 4, 5 });
                DVVector dStencil = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
                DVVector dOut = new DVVector("int32_t", 6);
                long count = TRTC.Copy_If_Stencil(dIn, dStencil, dOut, is_even);
                print_array((int[])dOut.to_host(0, count));
            }
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
