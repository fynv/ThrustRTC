using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_custom_vector
{
    class test_custom_vector
    {
        static void Main(string[] args)
        {
            DVVector d_in = new DVVector(new int[] { 0, 1, 2, 3, 4 });
            DVCustomVector src = new DVCustomVector(new DeviceViewable[] { d_in }, new string[] { "src" }, "idx",
                "        return src[idx % src.size()];\n", "int32_t", d_in.size() * 5);
            DVVector dst= new DVVector("int32_t", 25);
            TRTC.Copy(src, dst);
            print_array((int[])dst.to_host());
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
