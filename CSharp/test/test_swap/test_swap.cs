using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_swap
{
    class test_swap
    {
        static void Main(string[] args)
        {
            DVVector darr1 = new DVVector(new int[] { 10, 20, 30, 40, 50, 60, 70, 80 });
            DVVector darr2 = new DVVector(new int[] { 1000, 900, 800, 700, 600, 500, 400, 300 });
            TRTC.Swap(darr1, darr2);
            print_array((int[])darr1.to_host());
            print_array((int[])darr2.to_host());
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
