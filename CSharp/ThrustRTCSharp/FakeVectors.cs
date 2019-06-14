using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCLR;
using System.Runtime.InteropServices;

namespace ThrustRTCSharp
{
    public class DVConstant : DVVectorLike
    {
        public DVConstant(DeviceViewable dvobj, long size = -1)
            : base(Native.dvconstant_create(dvobj.m_cptr, (ulong)size))
        {
            m_dvobj = dvobj;
        }
        private readonly DeviceViewable m_dvobj;
    }

    public class DVCounter : DVVectorLike
    {
        public DVCounter(DeviceViewable dvobj_init, long size = -1)
            : base(Native.dvcounter_create(dvobj_init.m_cptr, (ulong)size))
        {
            m_dvobj_init = dvobj_init;
        }
        private readonly DeviceViewable m_dvobj_init;
    }

    public class DVDiscard : DVVectorLike
    {
        public DVDiscard(string elem_cls, long size = -1)
            : base(Native.dvdiscard_create(Marshal.StringToHGlobalAnsi(elem_cls), (ulong)size))  {}
    }

    public class DVPermutation : DVVectorLike
    {
        public DVPermutation(DVVectorLike vec_value, DVVectorLike vec_index)
            : base(Native.dvpermutation_create(vec_value.m_cptr, vec_index.m_cptr))
        {
            m_vec_value = vec_value;
            m_vec_index = vec_index;
        }
        private readonly DVVectorLike m_vec_value;
        private readonly DVVectorLike m_vec_index;
    }

    public class DVReverse : DVVectorLike
    {
        public DVReverse(DVVectorLike vec_value)
            : base(Native.dvreverse_create(vec_value.m_cptr))
        {
            m_vec_value = vec_value;
        }
        private readonly DVVectorLike m_vec_value;
    }

    public class DVTransform : DVVectorLike
    {
        public DVTransform(DVVectorLike vec_in, string elem_cls, Functor op)
            : base(Native.dvtransform_create(vec_in.m_cptr, Marshal.StringToHGlobalAnsi(elem_cls), op.m_cptr) )
        {
            m_vec_in = vec_in;
            m_op = op;
        }
        private readonly DVVectorLike m_vec_in;
        private readonly Functor m_op;
    }

    public class DVZipped : DVVectorLike
    {
        static IntPtr create(DVVectorLike[] vecs, string[] elem_names)
        {
            IntPtr[] p_vecs = new IntPtr[vecs.Length];
            for (int i = 0; i < vecs.Length; i++)
                p_vecs[i] = vecs[i].m_cptr;
            IntPtr[] p_elem_names = new IntPtr[elem_names.Length];
            for (int i = 0; i < elem_names.Length; i++)
                p_elem_names[i] = Marshal.StringToHGlobalAnsi(elem_names[i]);

            return Native.dvzipped_create(p_vecs, p_elem_names);
        }

        public DVZipped(DVVectorLike[] vecs, string[] elem_names)
            : base (create(vecs, elem_names))
        {
            m_vecs = vecs;
        }

        private readonly DVVectorLike[] m_vecs;
    }
}
