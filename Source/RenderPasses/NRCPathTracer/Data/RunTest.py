def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("NRCPathTracer.dll")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    # ToneMappingPass2 = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    # ToneMappingPass3 = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "ToneMapped")
    g.addPass(ToneMappingPass, "NRCToneMapped")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")
    NRCPathTracer = createPass("NRCPathTracer", {'params': PathTracerParams(useVBuffer=0)})
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
    g.addEdge("NRCPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapped.src")
    g.addEdge("NRCPathTracer.result", "NRCToneMapped.src")
    # g.markOutput("ToneMapped.dst")
    g.markOutput("NRCToneMapped.dst")
    # g.markOutput("NRCPathTracer.result")
    # g.markOutput("NRCPathTracer.factor")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None

# the root directory is the location of Mogwai executable.

# to assign an auto-quit frame.
# m.clock.exitFrame = 101

m.loadScene('../../../Media/Arcade/Arcade.pyscene', buildFlags=SceneBuilderFlags.Default)

# m.frameCapture.outputDir = '../../../Outputs/Image'
# m.frameCapture.baseFilename = 'arcade_nrc'
# m.frameCapture.ui = False
# m.frameCapture.addFrames(m.activeGraph, range(10))

# m.clock.pause()
# for i in range(10):
# 	renderFrame()
# 	m.frameCapture.baseFilename = f"arcade_nrc_{i:04d}"
#   m.frameCapture.capture()
# exit()

m.videoCapture.outputDir = '../../../Outputs/Video'
m.videoCapture.ui = False
m.videoCapture.baseFilename = 'arcade_nrc'
m.videoCapture.codec = Codec.Raw
m.videoCapture.fps = 24
m.videoCapture.addRanges(m.activeGraph, [[0, 300]])