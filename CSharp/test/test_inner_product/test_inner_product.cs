using System;
using ThrustRTCSharp;

namespace test_inner_product
{
    class test_inner_product
    {
        static void Main(string[] args)
        {
            DVVector d_vec1 = new DVVector(new float[] { 1.0f, 2.0f, 5.0f });
            DVVector d_vec2 = new DVVector(new float[] { 4.0f, 1.0f, 5.0f });
            Console.WriteLine(TRTC.Inner_Product(d_vec1, d_vec2, new DVFloat(0.0f)));
            Console.WriteLine(TRTC.Inner_Product(d_vec1, d_vec2, new DVFloat(0.0f), new Functor("Plus"), new Functor("Multiplies")));
        }
    }
}
