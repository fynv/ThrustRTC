using ThrustRTCSharp;

namespace test_discard
{
    class test_discard
    {
        static void Main(string[] args)
        {
            TRTC.Set_Verbose();

            // just to verify that it compiles
            DVCounter src = new DVCounter(new DVInt32(5), 10);
            DVDiscard sink = new DVDiscard("int32_t");
            TRTC.Copy(src, sink);

        }
    }
}
