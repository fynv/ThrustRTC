using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ThrustRTCSharp;

namespace test_count
{
    class test_count
    {
        static void Main(string[] args)
        {
            int[] hin=new int[2000];
            for (int i = 0; i < 2000; i++)
                hin[i] = i % 100;

            DVVector din = new DVVector(hin);
            Console.WriteLine(TRTC.Count(din, new DVInt32(47)));

            TRTC.Sequence(din);
            Functor op = new Functor(new string[]{ "x" }, "        return (x%100)==47;\n" );
            Console.WriteLine(TRTC.Count_If(din, op));

        }

        
    }
}
