using System;
using ThrustRTCSharp;

namespace test_unique
{
    class test_unique
    {
        static void Main(string[] args)
        {
            {
                DVVector d_value = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                long count = TRTC.Unique(d_value);
                print_array((int[])d_value.to_host(0, count));
            }

            {
                DVVector d_value = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                long count = TRTC.Unique(d_value, new Functor("EqualTo"));
                print_array((int[])d_value.to_host(0, count));
            }

            {
                DVVector d_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_out = new DVVector("int32_t", 7);
                long count = TRTC.Unique_Copy(d_in, d_out);
                print_array((int[])d_out.to_host(0, count));
            }

            {
                DVVector d_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_out = new DVVector("int32_t", 7);
                long count = TRTC.Unique_Copy(d_in, d_out, new Functor("EqualTo"));
                print_array((int[])d_out.to_host(0, count));
            }

            {
                DVVector d_keys = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_values = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
                long count = TRTC.Unique_By_Key(d_keys, d_values);
                print_array((int[])d_keys.to_host(0, count));
                print_array((int[])d_values.to_host(0, count));
            }

            {
                DVVector d_keys = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_values = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
                long count = TRTC.Unique_By_Key(d_keys, d_values, new Functor("EqualTo"));
                print_array((int[])d_keys.to_host(0, count));
                print_array((int[])d_values.to_host(0, count));
            }

            {
                DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
                DVVector d_keys_out = new DVVector("int32_t", 7);
                DVVector d_values_out = new DVVector("int32_t", 7);
                long count = TRTC.Unique_By_Key_Copy(d_keys_in, d_values_in, d_keys_out, d_values_out);
                print_array((int[])d_keys_out.to_host(0, count));
                print_array((int[])d_values_out.to_host(0, count));
            }

            {
                DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
                DVVector d_keys_out = new DVVector("int32_t", 7);
                DVVector d_values_out = new DVVector("int32_t", 7);
                long count = TRTC.Unique_By_Key_Copy(d_keys_in, d_values_in, d_keys_out, d_values_out, new Functor("EqualTo"));
                print_array((int[])d_keys_out.to_host(0, count));
                print_array((int[])d_values_out.to_host(0, count));
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
