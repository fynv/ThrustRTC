using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_trtc
{
    class test_trtc
    {
        static void Main(string[] args)
        {
            Kernel ker = new Kernel(new String[]{ "arr_in", "arr_out", "k" }, 
@"
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= arr_in.size()) return;
    arr_out[idx] = arr_in[idx]*k;");

            float[] test_f = new float[]{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            DVVector dvec_in_f = new DVVector(test_f);
            DVVector dvec_out_f = new DVVector("float", 5);
            DVFloat k1 = new DVFloat(10.0f);
            DeviceViewable[] args_f = new DeviceViewable[] { dvec_in_f, dvec_out_f, k1 };
            ker.launch(1, 128, args_f);
            float[] output = (float[])dvec_out_f.to_host();

            foreach (var item in output)
            {
                Console.WriteLine(item.ToString());
            }

        }
    }
}
