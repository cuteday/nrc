## NRC

This is a naive and dirty implementation of the paper Real-time Neural Radiance Caching for Path Tracing. The main implementation is located at [here](Source/RenderPasses/NRCPathTracer). The ReSTIR part described in the paper is not implemented.

These are some really old code implemented on an old versiton of Falcor, just for fun. The performance is slow due to my poor implementation. The noise of the 1-spp PT has reduced, but I have no confidence on the correctness of my implementation \_(:з」∠)\_.

<center>    <img style="width: 缩放比例; border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="demo.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;         display: inline-block; color: #999; padding: 2px;">Rendered at 1spp. Left: Default PT; Right: NRC.</div> </center>

