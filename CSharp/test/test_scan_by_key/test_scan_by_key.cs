using System;
using ThrustRTCSharp;

namespace test_scan_by_key
{
    class test_scan_by_key
    {
        static void Main(string[] args)
        {
            {
                DVVector d_keys = new DVVector(new int[] { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 });
                DVVector d_values = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
                TRTC.Inclusive_Scan_By_Key(d_keys, d_values, d_values);
                print_array((int[])d_values.to_host());
            }

            {
                DVVector d_keys = new DVVector(new int[] { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 });
                DVVector d_values = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
                TRTC.Exclusive_Scan_By_Key(d_keys, d_values, d_values);
                print_array((int[])d_values.to_host());
            }

            {
                DVVector d_keys = new DVVector(new int[] { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 });
                DVVector d_values = new DVVector(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
                TRTC.Exclusive_Scan_By_Key(d_keys, d_values, d_values, new DVInt32(5));
                print_array((int[])d_values.to_host());
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
