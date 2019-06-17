using System;
using ThrustRTCSharp;

namespace test_sequence
{
    class test_sequence
    {
        static void Main(string[] args)
        {
            DVVector vec= new DVVector("int32_t", 10);

            TRTC.Sequence(vec);
            print_array((int[])vec.to_host());

            TRTC.Sequence(vec, new DVInt32(1));
            print_array((int[])vec.to_host());

            TRTC.Sequence(vec, new DVInt32(1), new DVInt32(3));
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
