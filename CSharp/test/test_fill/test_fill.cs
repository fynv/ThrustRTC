using System;
using ThrustRTCSharp;

namespace test_fill
{
    class test_fill
    {
        static void Main(string[] args)
        {
            DVVector vec_to_fill = new DVVector("int32_t", 5);
            TRTC.Fill(vec_to_fill, new DVInt32(123));
            print_array((int[])vec_to_fill.to_host());
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
