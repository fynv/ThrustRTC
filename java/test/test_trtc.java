import java.util.*;
import JThrustRTC.*;

public class test_trtc 
{
	public static void main(String[] args) 
	{

	    Kernel ker = new Kernel(new String[]{ "arr_in", "arr_out", "k" }, 
	   	String.join("",
	  	"    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n",
	    "    if (idx >= arr_in.size()) return;\n",
	    "    arr_out[idx] = arr_in[idx]*k;\n"));

	    DVVector dvec_in_f = new DVVector(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

	    {
	    	float[] in = (float[])dvec_in_f.to_host();
	    	System.out.println(Arrays.toString(in));
	    }

	    DVVector dvec_out_f = new DVVector("float", 5);
	    DVFloat k1 = new DVFloat(10.0f);
	    DeviceViewable[] args_f = new DeviceViewable[] { dvec_in_f, dvec_out_f, k1 };
	    ker.launch(1, 128, args_f, 0);

	    System.out.println(Arrays.toString((float[])dvec_out_f.to_host()));

	}
}
