package JThrustRTC;
import java.lang.ref.Cleaner;

public class TRTC 
{
	static final Cleaner cleaner = Cleaner.create();
	
	public static void set_libnvrtc_path(String path)
	{
		Native.set_libnvrtc_path(path);
	}

	public static void Set_Verbose(boolean verbose)
	{
		Native.set_verbose(verbose);
	}

	public static void Set_Verbose()
	{
		Native.set_verbose(true);
	}

	public static void Add_Include_Dir(String dir)
	{
		Native.add_include_dir(dir);
	}

	public static void Add_Built_In_Header(String filename, String filecontent)
	{
		Native.add_built_in_header(filename, filecontent);
	}

	public static void Add_Inlcude_Filename(String filename)
	{
		Native.add_include_filename(filename);
	}

	public static void Add_Code_Block(String code)
    {
        Native.add_code_block(code);
    }

    // Transformations
    public static boolean Fill(DVVectorLike vec, DeviceViewable value)
    {
    	return Native.fill(vec.cptr(), value.cptr());
    }
}
