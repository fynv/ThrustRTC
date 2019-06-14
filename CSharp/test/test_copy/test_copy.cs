using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_copy
{
    class test_copy
    {
        static void Main(string[] args)
        {
            {
                DVVector dIn = new DVVector(new int[] { 10, 20, 30, 40, 50, 60, 70, 80 });
                DVVector dOut = new DVVector("int32_t", 8);
                TRTC.Copy(dIn, dOut);
                print_array((int[])dOut.to_host());
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
