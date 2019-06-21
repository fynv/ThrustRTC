using System;
using ThrustRTCSharp;

namespace test_trtc
{
    class test_trtc
    {
        static void Main(string[] args)
        {
            Kernel ker = new Kernel(new string[]{ "arr_in", "arr_out", "k" }, 
@"
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= arr_in.size()) return;
    arr_out[idx] = arr_in[idx]*k;");

            DVVector dvec_in_f = new DVVector(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
            DVVector dvec_out_f = new DVVector("float", 5);
            DVFloat k1 = new DVFloat(10.0f);
            DeviceViewable[] args_f = new DeviceViewable[] { dvec_in_f, dvec_out_f, k1 };
            ker.launch(1, 128, args_f);
            print_array((float[])dvec_out_f.to_host());

            DVVector dvec_in_i = new DVVector(new int[] { 6, 7, 8, 9, 10 });
            DVVector dvec_out_i = new DVVector("int32_t", 5);
            DVInt32 k2 = new DVInt32(5);
            DeviceViewable[] args_i = new DeviceViewable[] { dvec_in_i, dvec_out_i, k2 };
            ker.launch(1, 128, args_i);
            print_array((int[])dvec_out_i.to_host());

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
