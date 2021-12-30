def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'autoReset': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 6.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Halton, 'sampleCount': 1024})
    g.addPass(GBufferRT, "GBufferRT")
    MegakernelPathTracer = createPass("MegakernelPathTracer", {'params': PathTracerParams(
        samplesPerPixel=1, useFixedSeed = False,
        useVBuffer=0, maxBounces=20, maxNonSpecularBounces=20, useRussianRoulette=True, probabilityAbsorption=0.15)})
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
    g.markOutput("AccumulatePass.output")
    g.markOutput("ToneMappingPass.dst")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None

param_n_accumulate_frames = 256
param_n_video_frames = 60
param_n_image_frames = 50
param_n_image_skip_frames = 30
param_exit_frame = 1000
param_base_filename = "zeroday_ref"

# m.loadScene('../../../Media/Arcade/Arcade.pyscene', buildFlags=SceneBuilderFlags.Default)
# m.loadScene('../../../Media/ORCA/ZeroDay_v1/MEASURE_ONE/zeroday.pyscene', buildFlags=SceneBuilderFlags.Default)
m.loadScene('../../../Media/ORCA/ZeroDay_v1/MEASURE_SEVEN/zeroday_colored.pyscene', buildFlags=SceneBuilderFlags.Default)

# configures clock and framerate so the timepoint of the frames we rendered are deterministic

m.clock.stop()	# pause AND reset the clock
m.clock.framerate = 60

########################################################################################
# The following code renders the first 10 frames, at a 24fps speed, as reference images.
# The images are converged through accumulating multiple frames.
########################################################################################

m.frameCapture.outputDir = '../../../Outputs/Image/GT'
m.frameCapture.baseFilename = param_base_filename
m.frameCapture.ui = False

for i in range(param_n_image_frames):
    # m.scene.animated = False
    # m.clock.pause()
    # m.clock.frame = i
    for j in range(param_n_accumulate_frames):
        m.renderFrame()
    m.frameCapture.baseFilename = f"{param_base_filename}_{i:04d}"
    m.frameCapture.capture()
    #m.scene.animated = True
    m.clock.step(frames=param_n_image_skip_frames)

########################################################################################
# The following code 
########################################################################################

# m.videoCapture.outputDir = '../../../Outputs/Video'
# m.videoCapture.ui = False
# m.videoCapture.baseFilename = param_base_filename
# m.videoCapture.codec = Codec.Raw
# m.videoCapture.fps = 60
# m.videoCapture.bitrate = 4.0
# m.videoCapture.gopSize = 10
# video_frame_ranges = [[10, 12]]
# m.videoCapture.addRanges(m.activeGraph, video_frame_ranges)

# for i in range(param_n_video_frames):
#     for j in range(param_n_accumulate_frames):
#         m.renderFrame()
#     m.clock.step(frames=1)

exit()