import JThrustRTC.*;

public class test_for_each
{
	public static void main(String[] args) 
	{
        Functor printf_functor = new Functor(new String[]{ "x" }, "        printf(\"%d\\n\", x);\n");
        DVVector vec = new DVVector(new int[] { 1, 2, 3, 1, 2 });
        TRTC.For_Each(vec, printf_functor);		
	}	

}
