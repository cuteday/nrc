def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")
    loadRenderPassLibrary('DebugPasses.dll')
    loadRenderPassLibrary("NRCPathTracer.dll")
    loadRenderPassLibrary('Utils.dll')

    SideBySidePass = createPass('SideBySidePass', {'splitLocation': -1.0, 'showTextLabels': False, 'leftLabel': 'Left side', 'rightLabel': 'Right side'})
    g.addPass(SideBySidePass, 'SideBySidePass')
    SplitScreenPass = createPass('SplitScreenPass', {'splitLocation': 0.5, 'showTextLabels': False, 'leftLabel': 'PT', 'rightLabel': 'NRC'})
    g.addPass(SplitScreenPass, 'SplitScreenPass')
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 6.0})
    g.addPass(ToneMappingPass, "ToneMapped")
    g.addPass(ToneMappingPass, "NRCToneMapped")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Halton, 'sampleCount': 1024})
    g.addPass(GBufferRT, "GBufferRT")
    NRCPathTracer = createPass("NRCPathTracer", {'params': PathTracerParams(useVBuffer=0, maxBounces=15, maxNonSpecularBounces=15)})
    MegakernelPathTracer = createPass("MegakernelPathTracer", {'params': PathTracerParams(useVBuffer=0)})
    g.addPass(MegakernelPathTracer, "MegakernelPathTracer")
    g.addPass(NRCPathTracer, "NRCPathTracer")

    g.addEdge("GBufferRT.vbuffer", "NRCPathTracer.vbuffer")      # Required by ray footprint.
    g.addEdge("GBufferRT.posW", "NRCPathTracer.posW")
    g.addEdge("GBufferRT.normW", "NRCPathTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "NRCPathTracer.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "NRCPathTracer.faceNormalW")
    g.addEdge("GBufferRT.viewW", "NRCPathTracer.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "NRCPathTracer.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "NRCPathTracer.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "NRCPathTracer.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "NRCPathTracer.mtlParams")
    g.addEdge("NRCPathTracer.result", "NRCToneMapped.src")

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
    g.addEdge("MegakernelPathTracer.color", "ToneMapped.src")

    g.addEdge("ToneMapped.dst", "SideBySidePass.leftInput")
    g.addEdge("NRCToneMapped.dst", "SideBySidePass.rightInput")
    g.addEdge("ToneMapped.dst", "SplitScreenPass.leftInput")
    g.addEdge("NRCToneMapped.dst", "SplitScreenPass.rightInput")

    g.markOutput("SideBySidePass.output")
    g.markOutput("SplitScreenPass.output")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None