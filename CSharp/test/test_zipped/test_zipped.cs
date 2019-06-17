using System;
using ThrustRTCSharp;

namespace test_zipped
{
    class test_zipped
    {
        static void Main(string[] args)
        {
            DVVector d_int_in = new DVVector(new int[] { 0, 1, 2, 3, 4 });
            DVVector d_float_in = new DVVector(new float[] { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f });

            DVVector d_int_out = new DVVector("int32_t", 5);
            DVVector d_float_out = new DVVector("float", 5);

            DVZipped src = new DVZipped(new DVVectorLike[] { d_int_in, d_float_in }, new string[] { "a", "b" });
            DVZipped dst = new DVZipped(new DVVectorLike[] { d_int_out, d_float_out }, new string[] { "a", "b" });

            TRTC.Copy(src, dst);

            print_array((int[])d_int_out.to_host());
            print_array((float[])d_float_out.to_host());
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
