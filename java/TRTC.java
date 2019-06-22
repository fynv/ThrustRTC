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

	public static boolean Replace(DVVectorLike vec, DeviceViewable old_value, DeviceViewable new_value)
	{
		return Native.replace(vec.cptr(), old_value.cptr(), new_value.cptr());
	}

	public static boolean Replace_If(DVVectorLike vec, Functor pred, DeviceViewable new_value)
	{
		return Native.replace_if(vec.cptr(), pred.cptr(), new_value.cptr());
	}

	public static boolean Replace_Copy(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable old_value, DeviceViewable new_value)
	{
		return Native.replace_copy(vec_in.cptr(), vec_out.cptr(), old_value.cptr(), new_value.cptr());
	}

	public static boolean Replace_Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred, DeviceViewable new_value)
	{
		return Native.replace_copy_if(vec_in.cptr(), vec_out.cptr(), pred.cptr(), new_value.cptr());
	}

	public static boolean For_Each(DVVectorLike vec, Functor f)
	{
		return Native.for_each(vec.cptr(), f.cptr());
	}

	public static boolean Adjacent_Difference(DVVectorLike vec_in, DVVectorLike vec_out)
	{
		return Native.adjacent_difference(vec_in.cptr(), vec_out.cptr());
	}

	public static boolean Adjacent_Difference(DVVectorLike vec_in, DVVectorLike vec_out, Functor f)
	{
		return Native.adjacent_difference(vec_in.cptr(), vec_out.cptr(), f.cptr());
	}

	public static boolean Sequence(DVVectorLike vec)
	{
		return Native.sequence(vec.cptr());
	}

	public static boolean Sequence(DVVectorLike vec, DeviceViewable value_init)
	{
		return Native.sequence(vec.cptr(), value_init.cptr());
	}

	public static boolean Sequence(DVVectorLike vec, DeviceViewable value_init, DeviceViewable value_step)
	{
		return Native.sequence(vec.cptr(), value_init.cptr(), value_step.cptr());
	}

	public static boolean Tabulate(DVVectorLike vec, Functor op)
	{
		return Native.tabulate(vec.cptr(), op.cptr());
	}

	public static boolean Transform(DVVectorLike vec_in, DVVectorLike vec_out, Functor op)
	{
		return Native.transform(vec_in.cptr(), vec_out.cptr(), op.cptr());
	}

	public static boolean Transform_Binary(DVVectorLike vec_in1, DVVectorLike vec_in2, DVVectorLike vec_out, Functor op)
	{
		return Native.transform_binary(vec_in1.cptr(), vec_in2.cptr(), vec_out.cptr(), op.cptr());
	}

	public static boolean Transform_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor op, Functor pred)
	{
		return Native.transform_if(vec_in.cptr(), vec_out.cptr(), op.cptr(), pred.cptr());
	}

	public static boolean Transform_If_Stencil(DVVectorLike vec_in, DVVectorLike vec_stencil,  DVVectorLike vec_out, Functor op, Functor pred)
	{
		return Native.transform_if_stencil(vec_in.cptr(), vec_stencil.cptr(), vec_out.cptr(), op.cptr(), pred.cptr());
	}

	public static boolean Transform_Binary_If_Stencil(DVVectorLike vec_in1, DVVectorLike vec_in2, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor op, Functor pred)
	{
		return Native.transform_binary_if_stencil(vec_in1.cptr(), vec_in2.cptr(), vec_stencil.cptr(), vec_out.cptr(), op.cptr(), pred.cptr());
	}

	// Copying
	public static boolean Gather(DVVectorLike vec_map, DVVectorLike vec_in, DVVectorLike vec_out)
	{
		return Native.gather(vec_map.cptr(), vec_in.cptr(), vec_out.cptr());
	}

	public static boolean Gather_If(DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_in, DVVectorLike vec_out)
	{
		return Native.gather_if(vec_map.cptr(), vec_stencil.cptr(), vec_in.cptr(), vec_out.cptr());
	}

	public static boolean Gather_If(DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_in, DVVectorLike vec_out, Functor pred)
	{
		return Native.gather_if(vec_map.cptr(), vec_stencil.cptr(), vec_in.cptr(), vec_out.cptr(), pred.cptr());
	}

	public static boolean Scatter(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_out)
	{
		return Native.scatter(vec_in.cptr(), vec_map.cptr(), vec_out.cptr());
	}

	public static boolean Scatter_If(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_out)
	{
		return Native.scatter_if(vec_in.cptr(), vec_map.cptr(), vec_stencil.cptr(), vec_out.cptr());
	}

	public static boolean Scatter_If(DVVectorLike vec_in, DVVectorLike vec_map, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor pred)
	{
		return Native.scatter_if(vec_in.cptr(), vec_map.cptr(), vec_stencil.cptr(), vec_out.cptr(), pred.cptr());
	}

	public static boolean Copy(DVVectorLike vec_in, DVVectorLike vec_out)
	{
		return Native.copy(vec_in.cptr(), vec_out.cptr());
	}

	public static boolean Swap(DVVectorLike vec1, DVVectorLike vec2)
	{
		return Native.swap(vec1.cptr(), vec2.cptr());
	}

	// Redutions
	public static int Count(DVVectorLike vec, DeviceViewable value)
	{
		return Native.count(vec.cptr(), value.cptr());
	}

	public static int Count_If(DVVectorLike vec, Functor pred)
	{
		return Native.count_if(vec.cptr(), pred.cptr());
	}

	public static Object Reduce(DVVectorLike vec)
	{
		return Native.reduce(vec.cptr());
	}

	public static Object Reduce(DVVectorLike vec, DeviceViewable init)
	{
		return Native.reduce(vec.cptr(), init.cptr());
	}

	public static Object Reduce(DVVectorLike vec, DeviceViewable init, Functor binary_op)
	{
		return Native.reduce(vec.cptr(), init.cptr(), binary_op.cptr());
	}

	public static int Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out)
	{
		return Native.reduce_by_key(key_in.cptr(), value_in.cptr(), key_out.cptr(), value_out.cptr());
	}

	public static int Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out, Functor binary_pred)
	{
		return Native.reduce_by_key(key_in.cptr(), value_in.cptr(), key_out.cptr(), value_out.cptr(), binary_pred.cptr());
	}

	public static int Reduce_By_Key(DVVectorLike key_in, DVVectorLike value_in, DVVectorLike key_out, DVVectorLike value_out, Functor binary_pred, Functor binary_op)
	{
		return Native.reduce_by_key(key_in.cptr(), value_in.cptr(), key_out.cptr(), value_out.cptr(), binary_pred.cptr(), binary_op.cptr());
	}

	public static Boolean Equal(DVVectorLike vec1, DVVectorLike vec2)
	{
		return Native.equal(vec1.cptr(), vec2.cptr());
	}

	public static Boolean Equal(DVVectorLike vec1, DVVectorLike vec2, Functor binary_pred)
	{
		return Native.equal(vec1.cptr(), vec2.cptr(), binary_pred.cptr());
	}

	public static int Min_Element(DVVectorLike vec)
	{
		return Native.min_element(vec.cptr());
	}

	public static int Min_Element(DVVectorLike vec, Functor comp)
	{
		return Native.min_element(vec.cptr(), comp.cptr());
	}

	public static int Max_Element(DVVectorLike vec)
	{
		return Native.max_element(vec.cptr());
	}

	public static int Max_Element(DVVectorLike vec, Functor comp)
	{
		return Native.max_element(vec.cptr(), comp.cptr());
	}

	public static int[] MinMax_Element(DVVectorLike vec)
	{
		return Native.minmax_element(vec.cptr());
	}

	public static int[] MinMax_Element(DVVectorLike vec, Functor comp)
	{
		return Native.minmax_element(vec.cptr(), comp.cptr());
	}

	public static Object Inner_Product(DVVectorLike vec1, DVVectorLike vec2, DeviceViewable init)
	{
		return Native.inner_product(vec1.cptr(), vec2.cptr(), init.cptr());
	}

	public static Object Inner_Product(DVVectorLike vec1, DVVectorLike vec2, DeviceViewable init, Functor binary_op1, Functor binary_op2)
	{
		return Native.inner_product(vec1.cptr(), vec2.cptr(), init.cptr(), binary_op1.cptr(), binary_op2.cptr());
	}

	public static Object Transform_Reduce(DVVectorLike vec, Functor unary_op, DeviceViewable init, Functor binary_op)
	{
		return Native.transform_reduce(vec.cptr(), unary_op.cptr(), init.cptr(), binary_op.cptr());
	}

	public static Boolean All_Of(DVVectorLike vec, Functor pred)
	{
		return Native.all_of(vec.cptr(), pred.cptr());
	}

	public static Boolean Any_Of(DVVectorLike vec, Functor pred)
	{
		return Native.any_of(vec.cptr(), pred.cptr());
	}

	public static Boolean None_Of(DVVectorLike vec, Functor pred)
	{
		return Native.none_of(vec.cptr(), pred.cptr());
	}

	public static Boolean Is_Partitioned(DVVectorLike vec, Functor pred)
	{
		return Native.is_partitioned(vec.cptr(), pred.cptr());
	}

	public static Boolean Is_Sorted(DVVectorLike vec)
	{
		return Native.is_sorted(vec.cptr());
	}

	public static Boolean Is_Sorted(DVVectorLike vec, Functor comp)
	{
		return Native.is_sorted(vec.cptr(), comp.cptr());
	}

	// PrefixSums
	public static boolean Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out)
	{
		return Native.inclusive_scan(vec_in.cptr(), vec_out.cptr());
	}

	public static boolean Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor binary_op)
	{
		return Native.inclusive_scan(vec_in.cptr(), vec_out.cptr(), binary_op.cptr());
	}

	public static boolean Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out)
	{
		return Native.exclusive_scan(vec_in.cptr(), vec_out.cptr());
	}

	public static boolean Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable init)
	{
		return Native.exclusive_scan(vec_in.cptr(), vec_out.cptr(), init.cptr());
	}

	public static boolean Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable init, Functor binary_op)
	{
		return Native.exclusive_scan(vec_in.cptr(), vec_out.cptr(), init.cptr(), binary_op.cptr());
	}

	public static boolean Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out)
	{
		return Native.inclusive_scan_by_key(vec_key.cptr(), vec_value.cptr(), vec_out.cptr());
	}

	public static boolean Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, Functor binary_pred)
	{
		return Native.inclusive_scan_by_key(vec_key.cptr(), vec_value.cptr(), vec_out.cptr(), binary_pred.cptr());
	}

	public static boolean Inclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, Functor binary_pred, Functor binary_op)
	{
		return Native.inclusive_scan_by_key(vec_key.cptr(), vec_value.cptr(), vec_out.cptr(), binary_pred.cptr(), binary_op.cptr());
	}

	public static boolean Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out)
	{
		return Native.exclusive_scan_by_key(vec_key.cptr(), vec_value.cptr(), vec_out.cptr());
	}

	public static boolean Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init)
	{
		return Native.exclusive_scan_by_key(vec_key.cptr(), vec_value.cptr(), vec_out.cptr(), init.cptr());
	}

	public static boolean Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init, Functor binary_pred)
	{
		return Native.exclusive_scan_by_key(vec_key.cptr(), vec_value.cptr(), vec_out.cptr(), init.cptr(), binary_pred.cptr());
	}

	public static boolean Exclusive_Scan_By_Key(DVVectorLike vec_key, DVVectorLike vec_value, DVVectorLike vec_out, DeviceViewable init, Functor binary_pred, Functor binary_op)
	{
		return Native.exclusive_scan_by_key(vec_key.cptr(), vec_value.cptr(), vec_out.cptr(), init.cptr(), binary_pred.cptr(), binary_op.cptr());
	}

	public static boolean Transform_Inclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor unary_op, Functor binary_op)
	{
		return Native.transform_inclusive_scan(vec_in.cptr(), vec_out.cptr(), unary_op.cptr(), binary_op.cptr());
	}

	public static boolean Transform_Exclusive_Scan(DVVectorLike vec_in, DVVectorLike vec_out, Functor unary_op, DeviceViewable init, Functor binary_op)
	{
		return Native.transform_exclusive_scan(vec_in.cptr(), vec_out.cptr(), unary_op.cptr(), init.cptr(), binary_op.cptr());
	}

	// Reordering
	public static int Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred)
	{
		return Native.copy_if(vec_in.cptr(), vec_out.cptr(), pred.cptr());
	}

	public static int Copy_If_Stencil(DVVectorLike vec_in, DVVectorLike vec_stencil, DVVectorLike vec_out, Functor pred)
	{
		return Native.copy_if_stencil(vec_in.cptr(), vec_stencil.cptr(), vec_out.cptr(), pred.cptr());
	}

	public static int Remove(DVVectorLike vec, DeviceViewable value)
	{
		return Native.remove(vec.cptr(), value.cptr());
	}

	public static int Remove_Copy(DVVectorLike vec_in, DVVectorLike vec_out, DeviceViewable value)
	{
		return Native.remove_copy(vec_in.cptr(), vec_out.cptr(), value.cptr());
	}

	public static int Remove_If(DVVectorLike vec, Functor pred)
	{
		return Native.remove_if(vec.cptr(), pred.cptr());
	}

	public static int Remove_Copy_If(DVVectorLike vec_in, DVVectorLike vec_out, Functor pred)
	{
		return Native.remove_copy_if(vec_in.cptr(), vec_out.cptr(), pred.cptr());
	}

	public static int Remove_If_Stencil(DVVectorLike vec, DVVectorLike stencil, Functor pred)
	{
		return Native.remove_if_stencil(vec.cptr(), stencil.cptr(), pred.cptr());
	}

	public static int Remove_Copy_If_Stencil(DVVectorLike vec_in, DVVectorLike stencil, DVVectorLike vec_out, Functor pred)
	{
		return Native.remove_copy_if_stencil(vec_in.cptr(), stencil.cptr(), vec_out.cptr(), pred.cptr());
	}

	public static int Unique(DVVectorLike vec)
	{
		return Native.unique(vec.cptr());
	}

	public static int Unique(DVVectorLike vec, Functor binary_pred)
	{
		return Native.unique(vec.cptr(), binary_pred.cptr());
	}

	public static int Unique_Copy(DVVectorLike vec_in, DVVectorLike vec_out)
	{
		return Native.unique_copy(vec_in.cptr(), vec_out.cptr());
	}

	public static int Unique_Copy(DVVectorLike vec_in, DVVectorLike vec_out, Functor binary_pred)
	{
		return Native.unique_copy(vec_in.cptr(), vec_out.cptr(), binary_pred.cptr());
	}

	public static int Unique_By_Key(DVVectorLike keys, DVVectorLike values)
	{
		return Native.unique_by_key(keys.cptr(), values.cptr());
	}

	public static int Unique_By_Key(DVVectorLike keys, DVVectorLike values, Functor binary_pred)
	{
		return Native.unique_by_key(keys.cptr(), values.cptr(), binary_pred.cptr());
	}

	public static int Unique_By_Key_Copy(DVVectorLike keys_in, DVVectorLike values_in, DVVectorLike keys_out, DVVectorLike values_out)
	{
		return Native.unique_by_key_copy(keys_in.cptr(), values_in.cptr(), keys_out.cptr(), values_out.cptr());
	}

	public static int Unique_By_Key_Copy(DVVectorLike keys_in, DVVectorLike values_in, DVVectorLike keys_out, DVVectorLike values_out, Functor binary_pred)
	{
		return Native.unique_by_key_copy(keys_in.cptr(), values_in.cptr(), keys_out.cptr(), values_out.cptr(), binary_pred.cptr());
	}

	public static int Partition(DVVectorLike vec, Functor pred)
	{
		return Native.partition(vec.cptr(), pred.cptr());
	}

	public static int Partition_Stencil(DVVectorLike vec, DVVectorLike stencil, Functor pred)
	{
		return Native.partition_stencil(vec.cptr(), stencil.cptr(), pred.cptr());
	}

	public static int Partition_Copy(DVVectorLike vec_in, DVVectorLike vec_true, DVVectorLike vec_false, Functor pred)
	{
		return Native.partition_copy(vec_in.cptr(), vec_true.cptr(), vec_false.cptr(), pred.cptr());
	}

	public static int Partition_Copy_Stencil(DVVectorLike vec_in, DVVectorLike stencil, DVVectorLike vec_true, DVVectorLike vec_false, Functor pred)
	{
		return Native.partition_copy_stencil(vec_in.cptr(), stencil.cptr(), vec_true.cptr(), vec_false.cptr(), pred.cptr());
	}

	// Searching
	public static Integer Find(DVVectorLike vec, DeviceViewable value)
	{
		return Native.find(vec.cptr(), value.cptr());
	}

	public static Integer Find_If(DVVectorLike vec, Functor pred)
	{
		return Native.find_if(vec.cptr(), pred.cptr());
	}

	public static Integer Find_If_Not(DVVectorLike vec, Functor pred)
	{
		return Native.find_if_not(vec.cptr(), pred.cptr());
	}

    public static Integer Mismatch(DVVectorLike vec1, DVVectorLike vec2)
    {
        return Native.mismatch(vec1.cptr(), vec2.cptr());
    }

    public static Integer Mismatch(DVVectorLike vec1, DVVectorLike vec2, Functor pred)
    {
        return Native.mismatch(vec1.cptr(), vec2.cptr(), pred.cptr());
    }

    public static Integer Lower_Bound(DVVectorLike vec, DeviceViewable value)
    {
        return Native.lower_bound(vec.cptr(), value.cptr());
    }

    public static Integer Lower_Bound(DVVectorLike vec, DeviceViewable value, Functor comp)
    {
        return Native.lower_bound(vec.cptr(), value.cptr(), comp.cptr());
    }

    public static Integer Upper_Bound(DVVectorLike vec, DeviceViewable value)
    {
        return Native.upper_bound(vec.cptr(), value.cptr());
    }

    public static Integer Upper_Bound(DVVectorLike vec, DeviceViewable value, Functor comp)
    {
        return Native.upper_bound(vec.cptr(), value.cptr(), comp.cptr());
    }

    public static Boolean Binary_Search(DVVectorLike vec, DeviceViewable value)
    {
        return Native.binary_search(vec.cptr(), value.cptr());
    }

    public static Boolean Binary_Search(DVVectorLike vec, DeviceViewable value, Functor comp)
    {
        return Native.binary_search(vec.cptr(), value.cptr(), comp.cptr());
    }

    public static boolean Lower_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result)
    {
        return Native.lower_bound_v(vec.cptr(), values.cptr(), result.cptr());
    }

    public static boolean Lower_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp)
    {
        return Native.lower_bound_v(vec.cptr(), values.cptr(), result.cptr(), comp.cptr());
    }

    public static boolean Upper_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result)
    {
        return Native.upper_bound_v(vec.cptr(), values.cptr(), result.cptr());
    }

    public static boolean Upper_Bound_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp)
    {
        return Native.upper_bound_v(vec.cptr(), values.cptr(), result.cptr(), comp.cptr());
    }

    public static boolean Binary_Search_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result)
    {
        return Native.binary_search_v(vec.cptr(), values.cptr(), result.cptr());
    }

    public static boolean Binary_Search_V(DVVectorLike vec, DVVectorLike values, DVVectorLike result, Functor comp)
    {
        return Native.binary_search_v(vec.cptr(), values.cptr(), result.cptr(), comp.cptr());
    }

    public static Integer Partition_Point(DVVectorLike vec, Functor pred)
    {
        return Native.partition_point(vec.cptr(), pred.cptr());
    }

    public static Integer Is_Sorted_Until(DVVectorLike vec)
    {
        return Native.is_sorted_until(vec.cptr());
    }

    public static Integer Is_Sorted_Until(DVVectorLike vec, Functor comp)
    {
        return Native.is_sorted_until(vec.cptr(), comp.cptr());
    }
}
