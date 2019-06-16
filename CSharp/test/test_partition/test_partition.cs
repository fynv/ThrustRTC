using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_partition
{
    class test_partition
    {
        static void Main(string[] args)
        {
            Functor is_even = new Functor( new string[]{ "x" }, "        return x % 2 == 0;\n" );

            {
                DVVector d_value = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
                TRTC.Partition(d_value, is_even);
                print_array((int[])d_value.to_host());
            }

            {
                DVVector d_value = new DVVector(new int[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
                DVVector d_stencil = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
                TRTC.Partition_Stencil(d_value, d_stencil, is_even);
                print_array((int[])d_value.to_host());
            }

            {
                DVVector d_value = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
                DVVector d_evens = new DVVector("int32_t", 10);
                DVVector d_odds = new DVVector("int32_t", 10);
                long count = TRTC.Partition_Copy(d_value, d_evens, d_odds, is_even);
                print_array((int[])d_evens.to_host(0, count));
                print_array((int[])d_odds.to_host(0, 10 - count));
            }

            {
                DVVector d_value = new DVVector(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
                DVVector d_stencil = new DVVector(new int[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
                DVVector d_evens = new DVVector("int32_t", 10);
                DVVector d_odds = new DVVector("int32_t", 10);
                long count = TRTC.Partition_Copy_Stencil(d_value, d_stencil, d_evens, d_odds, new Functor("Identity"));
                print_array((int[])d_evens.to_host(0, count));
                print_array((int[])d_odds.to_host(0, 10 - count));
            }

            {
                DVVector d_value = new DVVector(new int[] { 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 });
                Console.WriteLine(TRTC.Partition_Point(d_value, is_even));
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
