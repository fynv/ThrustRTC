using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_remove
{
    class test_remove
    {
        static void Main(string[] args)
        {
            {
                DVVector d_value = new DVVector(new int[] { 3, 1, 4, 1, 5, 9 });
                long count = TRTC.Remove(d_value, new DVInt32(1));
                print_array((int[])d_value.to_host(0, count));
            }

            {
                DVVector d_in = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
                DVVector d_out = new DVVector("int32_t", 6);
                long count = TRTC.Remove_Copy(d_in, d_out, new DVInt32(0));
                print_array((int[])d_out.to_host(0, count));
            }

            Functor is_even = new Functor(new string[] { "x" }, "        return x % 2 == 0;\n");

            {
                DVVector d_value = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
                long count = TRTC.Remove_If(d_value, is_even);
                print_array((int[])d_value.to_host(0, count));
            }

            {
                DVVector d_in = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
                DVVector d_out = new DVVector("int32_t", 6);
                long count = TRTC.Remove_Copy_If(d_in, d_out, is_even);
                print_array((int[])d_out.to_host(0, count));
            }

            Functor identity = new Functor("Identity");

            {
                DVVector d_value = new DVVector(new int[] { 1, 4, 2, 8, 5, 7 });
                DVVector d_stencil = new DVVector(new int[] { 0, 1, 1, 1, 0, 0 });
                long count = TRTC.Remove_If_Stencil(d_value, d_stencil, identity);
                print_array((int[])d_value.to_host(0, count));
            }

            {
                DVVector d_in = new DVVector(new int[] { -2, 0, -1, 0, 1, 2 });
                DVVector d_stencil = new DVVector(new int[] { 1, 1, 0, 1, 0, 1 });
                DVVector d_out = new DVVector("int32_t", 6);
                long count = TRTC.Remove_Copy_If_Stencil(d_in, d_stencil, d_out, identity);
                print_array((int[])d_out.to_host(0, count));
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
