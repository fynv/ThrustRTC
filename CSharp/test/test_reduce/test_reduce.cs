using System;
using ThrustRTCSharp;

namespace test_reduce
{
    class test_reduce
    {
        static void Main(string[] args)
        {
            {
                DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
                Console.WriteLine(TRTC.Reduce(darr));
                Console.WriteLine(TRTC.Reduce(darr, new DVInt32(1)));
                Console.WriteLine(TRTC.Reduce(darr, new DVInt32(-1), new Functor("Maximum")));
            }

            {
                DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
                DVVector d_keys_out = new DVVector("int32_t", 7);
                DVVector d_values_out = new DVVector("int32_t", 7);
                long count = TRTC.Reduce_By_Key(d_keys_in, d_values_in, d_keys_out, d_values_out);
                print_array((int[])d_keys_out.to_host(0, count));
                print_array((int[])d_values_out.to_host(0, count));
            }

            {
                DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
                DVVector d_keys_out = new DVVector("int32_t", 7);
                DVVector d_values_out = new DVVector("int32_t", 7);
                long count = TRTC.Reduce_By_Key(d_keys_in, d_values_in, d_keys_out, d_values_out, new Functor("EqualTo"));
                print_array((int[])d_keys_out.to_host(0, count));
                print_array((int[])d_values_out.to_host(0, count));
            }

            {
                DVVector d_keys_in = new DVVector(new int[] { 1, 3, 3, 3, 2, 2, 1 });
                DVVector d_values_in = new DVVector(new int[] { 9, 8, 7, 6, 5, 4, 3 });
                DVVector d_keys_out = new DVVector("int32_t", 7);
                DVVector d_values_out = new DVVector("int32_t", 7);
                long count = TRTC.Reduce_By_Key(d_keys_in, d_values_in, d_keys_out, d_values_out, new Functor("EqualTo"), new Functor("Plus"));
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
