import java.util.*;
import JThrustRTC.*;

public class test_transform_iter
{
	public static void main(String[] args) 
	{
		DVVector dvalues = new DVVector(new float[] { 1.0f, 4.0f, 9.0f, 16.0f });
		Functor square_root = new Functor(new String[] { "x" }, "        return sqrtf(x);\n");
		DVTransform src = new DVTransform(dvalues, "float", square_root);
		DVVector dst = new DVVector("float", 4);
		TRTC.Copy(src, dst);
		System.out.println(Arrays.toString((float[])dst.to_host()));
	}

}
