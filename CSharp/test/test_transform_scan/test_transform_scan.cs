using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_transform_scan
{
    class test_transform_scan
    {
        static void Main(string[] args)
        {
            {
                DVVector d_data = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
                TRTC.Transform_Inclusive_Scan(d_data, d_data, new Functor("Negate"), new Functor("Plus"));
                print_array((int[])d_data.to_host());
            }

            {
                DVVector d_data = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
                TRTC.Transform_Exclusive_Scan(d_data, d_data, new Functor("Negate"), new DVInt32(4), new Functor("Plus"));
                print_array((int[])d_data.to_host());
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
