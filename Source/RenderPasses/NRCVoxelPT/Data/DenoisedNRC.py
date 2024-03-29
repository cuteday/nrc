def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("NRCVoxelPT.dll")
    loadRenderPassLibrary("OptixDenoiser.dll")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 6.0})
    g.addPass(ToneMappingPass, "NRCToneMapped")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")
    NRCVoxelPT = createPass("NRCVoxelPT", {'params': PathTracerParams(useVBuffer=0, maxBounces=15, maxNonSpecularBounces=15)})
    g.addPass(NRCVoxelPT, "NRCVoxelPT")
    OptixDenoiser = createPass("OptixDenoiser")
    g.addPass(OptixDenoiser, "OptixDenoiser")

    g.addEdge("GBufferRT.vbuffer", "NRCVoxelPT.vbuffer")      # Required by ray footprint.
    g.addEdge("GBufferRT.posW", "NRCVoxelPT.posW")
    g.addEdge("GBufferRT.normW", "NRCVoxelPT.normalW")
    g.addEdge("GBufferRT.tangentW", "NRCVoxelPT.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "NRCVoxelPT.faceNormalW")
    g.addEdge("GBufferRT.viewW", "NRCVoxelPT.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "NRCVoxelPT.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "NRCVoxelPT.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "NRCVoxelPT.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "NRCVoxelPT.mtlParams")
    g.addEdge("NRCVoxelPT.result", "NRCToneMapped.src")
    g.addEdge("NRCToneMapped.dst", "OptixDenoiser.color")
    g.addEdge("NRCVoxelPT.albedo", "OptixDenoiser.albedo")
    g.addEdge("GBufferRT.normW", "OptixDenoiser.normal")
    g.markOutput("OptixDenoiser.output")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None
