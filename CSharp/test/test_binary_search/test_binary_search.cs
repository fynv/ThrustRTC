using System;
using ThrustRTCSharp;

namespace test_binary_search
{
    class test_binary_search
    {
        static void Main(string[] args)
        {
            DVVector d_input = new DVVector(new int[] { 0, 2, 5, 7, 8 });
            {
                Console.WriteLine(TRTC.Lower_Bound(d_input, new DVInt32(0)));
                Console.WriteLine(TRTC.Lower_Bound(d_input, new DVInt32(1)));
                Console.WriteLine(TRTC.Lower_Bound(d_input, new DVInt32(2)));
                Console.WriteLine(TRTC.Lower_Bound(d_input, new DVInt32(3)));
                Console.WriteLine(TRTC.Lower_Bound(d_input, new DVInt32(8)));
                Console.WriteLine(TRTC.Lower_Bound(d_input, new DVInt32(9)));
            }
            Console.WriteLine("");
            {
                Console.WriteLine(TRTC.Upper_Bound(d_input, new DVInt32(0)));
                Console.WriteLine(TRTC.Upper_Bound(d_input, new DVInt32(1)));
                Console.WriteLine(TRTC.Upper_Bound(d_input, new DVInt32(2)));
                Console.WriteLine(TRTC.Upper_Bound(d_input, new DVInt32(3)));
                Console.WriteLine(TRTC.Upper_Bound(d_input, new DVInt32(8)));
                Console.WriteLine(TRTC.Upper_Bound(d_input, new DVInt32(9)));
            }
            Console.WriteLine("");
            {
                Console.WriteLine(TRTC.Binary_Search(d_input, new DVInt32(0)));
                Console.WriteLine(TRTC.Binary_Search(d_input, new DVInt32(1)));
                Console.WriteLine(TRTC.Binary_Search(d_input, new DVInt32(2)));
                Console.WriteLine(TRTC.Binary_Search(d_input, new DVInt32(3)));
                Console.WriteLine(TRTC.Binary_Search(d_input, new DVInt32(8)));
                Console.WriteLine(TRTC.Binary_Search(d_input, new DVInt32(9)));
            }
            Console.WriteLine("");

            DVVector d_values = new DVVector(new int[] { 0, 1, 2, 3, 8, 9 });
            DVVector d_output = new DVVector("int32_t", 6);

            TRTC.Lower_Bound_V(d_input, d_values, d_output);
            print_array((int[])d_output.to_host());

            TRTC.Upper_Bound_V(d_input, d_values, d_output);
            print_array((int[])d_output.to_host());

            TRTC.Binary_Search_V(d_input, d_values, d_output);
            print_array((int[])d_output.to_host());

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
