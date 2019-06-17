using System;
using ThrustRTCSharp;

namespace test_gather
{
    class test_gather
    {
        static void Main(string[] args)
        {
            {
                DVVector d_values = new DVVector(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
                DVVector d_map = new DVVector(new int[] { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 });
                DVVector d_output = new DVVector("int32_t", 10);
                TRTC.Gather(d_map, d_values, d_output);
                print_array((int[])d_output.to_host());
            }

            {
                DVVector d_values = new DVVector(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
                DVVector d_stencil = new DVVector(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
                DVVector d_map = new DVVector(new int[] { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 });
                DVVector d_output = new DVVector("int32_t", 10);
                TRTC.Fill(d_output, new DVInt32(7));
                TRTC.Gather_If(d_map, d_stencil, d_values, d_output);
                print_array((int[])d_output.to_host());
            }

            Functor is_even = new Functor( new string[] { "x" }, "        return ((x % 2) == 0);\n" );
            {
                DVVector d_values = new DVVector(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
                DVVector d_stencil = new DVVector(new int[] { 0, 3, 4, 1, 4, 1, 2, 7, 8, 9 });
                DVVector d_map = new DVVector(new int[] { 0, 2, 4, 6, 8, 1, 3, 5, 7, 9 });
                DVVector d_output = new DVVector("int32_t", 10);
                TRTC.Fill(d_output, new DVInt32(7));
                TRTC.Gather_If(d_map, d_stencil, d_values, d_output, is_even);
                print_array((int[])d_output.to_host());
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
