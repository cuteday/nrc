def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")
    MegakernelPathTracer = createPass("MegakernelPathTracer", {'params': PathTracerParams(
		samplesPerPixel=256, useFixedSeed = True,
		useVBuffer=0, maxBounces=100, maxNonSpecularBounces=100, useRussianRoulette=True, probabilityAbsorption=0.2)})
    g.addPass(MegakernelPathTracer, "MegakernelPathTracer")
    g.addEdge("GBufferRT.vbuffer", "MegakernelPathTracer.vbuffer")      # Required by ray footprint.
    g.addEdge("GBufferRT.posW", "MegakernelPathTracer.posW")
    g.addEdge("GBufferRT.normW", "MegakernelPathTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "MegakernelPathTracer.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "MegakernelPathTracer.faceNormalW")
    g.addEdge("GBufferRT.viewW", "MegakernelPathTracer.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "MegakernelPathTracer.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "MegakernelPathTracer.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "MegakernelPathTracer.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "MegakernelPathTracer.mtlParams")
    g.addEdge("MegakernelPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMappingPass.src")
    g.markOutput("ToneMappingPass.dst")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None