def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("NRCPathTracer.dll")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 6.0})
    g.addPass(ToneMappingPass, "ToneMapped")
    g.addPass(ToneMappingPass, "NRCToneMapped")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Halton, 'sampleCount': 1024})
    g.addPass(GBufferRT, "GBufferRT")
    NRCPathTracer = createPass("NRCPathTracer", {'params': PathTracerParams(useVBuffer=0, maxBounces=15, maxNonSpecularBounces=15)})
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
    g.markOutput("NRCPathTracer.result")
    g.markOutput("NRCToneMapped.dst")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None

param_n_train_frames = 1024
param_n_accumulate_frames = 256
param_n_video_frames = 60
param_n_image_frames = 50
param_n_image_skip_frames = 30
param_exit_frame = 1000
param_base_filename = "zeroday_nrc"

# m.loadScene('../../../Media/Arcade/Arcade.pyscene', buildFlags=SceneBuilderFlags.Default)
# m.loadScene('../../../Media/ORCA/ZeroDay_v1/MEASURE_ONE/zeroday.pyscene', buildFlags=SceneBuilderFlags.Default)
m.loadScene('../../../Media/ORCA/ZeroDay_v1/MEASURE_SEVEN/zeroday_colored.pyscene', buildFlags=SceneBuilderFlags.Default)

########################################################################################
# We firstly give the NRC a warm-up, that is, train the network for about ~10s,
# after it converges, run the inference at given viewpoints.
########################################################################################
m.clock.framerate = 60
for _ in range(param_n_train_frames):
    m.renderFrame()

# configures clock and framerate so the timepoint of the frames we rendered are deterministic

m.clock.stop()	# pause AND reset the clock

########################################################################################
# The following code renders the first 10 frames, at a 24fps speed, as reference images.
# The images are converged through accumulating multiple frames.
########################################################################################

m.frameCapture.outputDir = '../../../Outputs/Image/NRC'
m.frameCapture.baseFilename = param_base_filename
m.frameCapture.ui = False

for i in range(param_n_image_frames * param_n_image_skip_frames):
    # m.scene.animated = False
    # m.clock.pause()
    # m.clock.frame = i
    m.renderFrame()
    if i % param_n_image_skip_frames == 0:
        idx = int(i / param_n_image_skip_frames)
        m.frameCapture.baseFilename = f"{param_base_filename}_{idx:04d}"
        m.frameCapture.capture()
        #m.scene.animated = True
    m.clock.step(frames=1)
