using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_replace
{
    class test_replace
    {
        static void Main(string[] args)
        {
            Functor is_less_than_zero = new Functor(new string[]{ "x" }, "        return x<0;\n" );

            {
                DVVector vec = new DVVector(new int[] { 1, 2, 3, 1, 2 });
                TRTC.Replace(vec, new DVInt32(1), new DVInt32(99));
                print_array((int[])vec.to_host());
            }

            {
                DVVector vec = new DVVector(new int[] { 1, -2, 3, -4, 5 });
                TRTC.Replace_If(vec, is_less_than_zero, new DVInt32(99));
                print_array((int[])vec.to_host());
            }

            {
                DVVector vec_in = new DVVector(new int[] { 1, 2, 3, 1, 2 });
                DVVector vec_out = new DVVector("int32_t", 5);
                TRTC.Replace_Copy(vec_in, vec_out, new DVInt32(1), new DVInt32(99));
                print_array((int[])vec_out.to_host());
            }

            {
                DVVector vec_in = new DVVector(new int[] { 1, -2, 3, -4, 5 });
                DVVector vec_out = new DVVector("int32_t", 5);
                TRTC.Replace_Copy_If(vec_in, vec_out, is_less_than_zero, new DVInt32(99));
                print_array((int[])vec_out.to_host());
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
