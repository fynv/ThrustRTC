using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_scatter
{
    class test_scatter
    {
        static void Main(string[] args)
        {
            {
                DVVector d_values = new DVVector(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
                DVVector d_map = new DVVector(new int[] { 0, 5, 1, 6, 2, 7, 3, 8, 4, 9 });
                DVVector d_output = new DVVector("int32_t", 10);
                TRTC.Scatter(d_values, d_map, d_output);
                print_array((int[])d_output.to_host());
            }

            {
                DVVector d_V = new DVVector(new int[] { 10, 20, 30, 40, 50, 60, 70, 80 });
                DVVector d_M = new DVVector(new int[] { 0, 5, 1, 6, 2, 7, 3, 4 });
                DVVector d_S = new DVVector(new int[] { 1, 0, 1, 0, 1, 0, 1, 0 });
                DVVector d_D = new DVVector(new int[] { 0, 0, 0, 0, 0, 0, 0, 0 });
                TRTC.Scatter_If(d_V, d_M, d_S, d_D);
                print_array((int[])d_D.to_host());
            }

            Functor is_even = new Functor(new string[] { "x" }, "        return ((x % 2) == 0);\n");
            {
                DVVector d_V = new DVVector(new int[] { 10, 20, 30, 40, 50, 60, 70, 80 });
                DVVector d_M = new DVVector(new int[] { 0, 5, 1, 6, 2, 7, 3, 4 });
                DVVector d_S = new DVVector(new int[] { 2, 1, 2, 1, 2, 1, 2, 1 });
                DVVector d_D = new DVVector(new int[] { 0, 0, 0, 0, 0, 0, 0, 0 });
                TRTC.Scatter_If(d_V, d_M, d_S, d_D, is_even);
                print_array((int[])d_D.to_host());
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
