using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_transform_iter
{
    class test_transform_iter
    {
        static void Main(string[] args)
        {
            DVVector dvalues = new DVVector(new float[] { 1.0f, 4.0f, 9.0f, 16.0f });
            Functor square_root = new Functor(new string[] { "x" }, "        return sqrtf(x);\n");
            DVTransform src = new DVTransform(dvalues, "float", square_root);
            DVVector dst = new DVVector("float", 4);
            TRTC.Copy(src, dst);
            print_array((float[])dst.to_host());

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
