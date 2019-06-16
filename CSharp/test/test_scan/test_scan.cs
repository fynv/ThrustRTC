using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_scan
{
    class test_scan
    {
        static void Main(string[] args)
        {
            {
                DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
                TRTC.Inclusive_Scan(darr, darr);
                print_array((int[])darr.to_host());
            }

            {
                DVVector darr = new DVVector(new int[] { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 });
                TRTC.Inclusive_Scan(darr, darr, new Functor("Maximum"));
                print_array((int[])darr.to_host());
            }

            {
                DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
                TRTC.Exclusive_Scan(darr, darr);
                print_array((int[])darr.to_host());
            }

            {
                DVVector darr = new DVVector(new int[] { 1, 0, 2, 2, 1, 3 });
                TRTC.Exclusive_Scan(darr, darr, new DVInt32(4));
                print_array((int[])darr.to_host());
            }

            {
                DVVector darr = new DVVector(new int[] { -5, 0, 2, -3, 2, 4, 0, -1, 2, 8 });
                TRTC.Exclusive_Scan(darr, darr, new DVInt32(1), new Functor("Maximum"));
                print_array((int[])darr.to_host());
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
