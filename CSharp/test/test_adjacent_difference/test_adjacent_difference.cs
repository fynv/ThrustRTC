using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_adjacent_difference
{
    class test_adjacent_difference
    {
        static void Main(string[] args)
        {
            {
                DVVector vec_in = new DVVector(new int[] { 1, 2, 1, 2, 1, 2, 1, 2 });
                DVVector vec_out = new DVVector("int32_t", 8);
                TRTC.Adjacent_Difference(vec_in, vec_out);
                print_array((int[])vec_out.to_host());
            }

            {
                DVVector vec_in = new DVVector(new int[] { 1, 2, 1, 2, 1, 2, 1, 2 });
                DVVector vec_out = new DVVector("int32_t", 8);
                TRTC.Adjacent_Difference(vec_in, vec_out, new Functor("Plus"));
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
