using System;
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

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_dvobj = null;
            base.Dispose(disposing);
        }

        private DeviceViewable m_dvobj;
    }

    public class DVCounter : DVVectorLike
    {
        public DVCounter(DeviceViewable dvobj_init, long size = -1)
            : base(Native.dvcounter_create(dvobj_init.m_cptr, (ulong)size))
        {
            m_dvobj_init = dvobj_init;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_dvobj_init = null;
            base.Dispose(disposing);
        }

        private DeviceViewable m_dvobj_init;
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

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_vec_value = null;
            m_vec_index = null;
            base.Dispose(disposing);
        }

        private DVVectorLike m_vec_value;
        private DVVectorLike m_vec_index;
    }

    public class DVReverse : DVVectorLike
    {
        public DVReverse(DVVectorLike vec_value)
            : base(Native.dvreverse_create(vec_value.m_cptr))
        {
            m_vec_value = vec_value;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_vec_value = null;
            base.Dispose(disposing);
        }

        private DVVectorLike m_vec_value;
    }

    public class DVTransform : DVVectorLike
    {
        public DVTransform(DVVectorLike vec_in, string elem_cls, Functor op)
            : base(Native.dvtransform_create(vec_in.m_cptr, Marshal.StringToHGlobalAnsi(elem_cls), op.m_cptr) )
        {
            m_vec_in = vec_in;
            m_op = op;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_vec_in = null;
            m_op = null;
            base.Dispose(disposing);
        }

        private DVVectorLike m_vec_in;
        private Functor m_op;
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

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_vecs = null;
            base.Dispose(disposing);
        }

        private DVVectorLike[] m_vecs;
    }

    public class DVCustomVector : DVVectorLike
    {
        static IntPtr create(DeviceViewable[] objs, string[] name_objs, string name_idx, string code_body, string elem_cls, long size, bool read_only)
        {
            CapturedDeviceViewable_clr[] arg_map = new CapturedDeviceViewable_clr[objs.Length];
            for (int i = 0; i < objs.Length; i++)
            {
                arg_map[i].obj = objs[i].m_cptr;
                arg_map[i].obj_name = Marshal.StringToHGlobalAnsi(name_objs[i]);
            }

            return Native.dvcustomvector_create(arg_map,
                Marshal.StringToHGlobalAnsi(name_idx),
                Marshal.StringToHGlobalAnsi(code_body),
                Marshal.StringToHGlobalAnsi(elem_cls), (ulong)size, read_only);
        }

        public DVCustomVector(DeviceViewable[] objs, string[] name_objs, string name_idx, string code_body, string elem_cls, long size = -1, bool read_only = true)
            : base(create(objs, name_objs, name_idx, code_body, elem_cls, size, read_only))
        {
            m_objs = objs;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposed) return;
            m_objs = null;
            base.Dispose(disposing);
        }

        private DeviceViewable[] m_objs;

    }
}
