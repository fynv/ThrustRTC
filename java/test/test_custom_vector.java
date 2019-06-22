import java.util.*;
import JThrustRTC.*;

public class test_custom_vector
{
	public static void main(String[] args) 
	{
		DVVector d_in = new DVVector(new int[] { 0, 1, 2, 3, 4 });
		DVCustomVector src = new DVCustomVector(new DeviceViewable[] { d_in }, new String[] { "src" }, "idx",
			"        return src[idx % src.size()];\n", "int32_t", d_in.size() * 5, true);
		DVVector dst= new DVVector("int32_t", 25);
		TRTC.Copy(src, dst);
		System.out.println(Arrays.toString((int[])dst.to_host()));
	}

}
