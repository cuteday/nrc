# # Graphs
# from falcor import *

# def render_graph_NRCVoxelPathTracerGraph():
#     g = RenderGraph('NRCVoxelPathTracerGraph')
#     loadRenderPassLibrary('NRCVoxelPT.dll')
#     loadRenderPassLibrary('AccumulatePass.dll')
#     loadRenderPassLibrary('ToneMapper.dll')
#     loadRenderPassLibrary('GBuffer.dll')
#     AccumulatePass = createPass('AccumulatePass', {'enabled': True, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0, 'maxAccumulatedFrames': 0})
#     g.addPass(AccumulatePass, 'AccumulatePass')
#     NRCToneMapped = createPass('ToneMapper', {'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': ExposureMode.AperturePriority})
#     g.addPass(NRCToneMapped, 'NRCToneMapped')
#     GBufferRT = createPass('GBufferRT', {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': CullMode.CullBack, 'texLOD': TexLODMode.Mip0, 'useTraceRayInline': False})
#     g.addPass(GBufferRT, 'GBufferRT')
#     NRCVoxelPT = createPass('NRCPathTracer', {'params': PathTracerParams(samplesPerPixel=1, lightSamplesPerVertex=1, maxBounces=15, maxNonSpecularBounces=15, useVBuffer=0, useAlphaTest=1, adjustShadingNormals=0, forceAlphaOne=1, clampSamples=0, clampThreshold=10.0, specularRoughnessThreshold=0.25, useBRDFSampling=1, useNEE=1, useMIS=1, misHeuristic=1, misPowerExponent=2.0, useRussianRoulette=0, probabilityAbsorption=0.20000000298023224, useFixedSeed=0, useNestedDielectrics=1, useLightsInDielectricVolumes=0, disableCaustics=0, rayFootprintMode=0, rayConeMode=2, rayFootprintUseRoughness=0), 'sampleGenerator': 1, 'emissiveSampler': EmissiveLightSamplerType.LightBVH, 'uniformSamplerOptions': LightBVHSamplerOptions(buildOptions=LightBVHBuilderOptions(splitHeuristicSelection=SplitHeuristic.BinnedSAOH, maxTriangleCountPerLeaf=10, binCount=16, volumeEpsilon=0.0010000000474974513, splitAlongLargest=False, useVolumeOverSA=False, useLeafCreationCost=True, createLeavesASAP=True, allowRefitting=True, usePreintegration=True, useLightingCones=True), useBoundingCone=True, useLightingCone=True, disableNodeFlux=False, useUniformTriangleSampling=True, solidAngleBoundMethod=SolidAngleBoundMethod.Sphere)})
#     g.addPass(NRCVoxelPT, 'NRCVoxelPT')
#     g.addEdge('GBufferRT.vbuffer', 'NRCVoxelPT.vbuffer')
#     g.addEdge('GBufferRT.posW', 'NRCVoxelPT.posW')
#     g.addEdge('GBufferRT.normW', 'NRCVoxelPT.normalW')
#     g.addEdge('GBufferRT.tangentW', 'NRCVoxelPT.tangentW')
#     g.addEdge('GBufferRT.faceNormalW', 'NRCVoxelPT.faceNormalW')
#     g.addEdge('GBufferRT.viewW', 'NRCVoxelPT.viewW')
#     g.addEdge('GBufferRT.diffuseOpacity', 'NRCVoxelPT.mtlDiffOpacity')
#     g.addEdge('GBufferRT.specRough', 'NRCVoxelPT.mtlSpecRough')
#     g.addEdge('GBufferRT.emissive', 'NRCVoxelPT.mtlEmissive')
#     g.addEdge('GBufferRT.matlExtra', 'NRCVoxelPT.mtlParams')
#     g.addEdge('NRCVoxelPT.result', 'AccumulatePass.input')
#     g.addEdge('AccumulatePass.output', 'NRCToneMapped.src')
#     g.markOutput('NRCToneMapped.dst')
#     return g
# m.addGraph(render_graph_NRCVoxelPathTracerGraph())

def render_graph_PathTracerGraph():
    g = RenderGraph("NRCVoxelPathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("NRCVoxelPT.dll")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "NRCToneMapped")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")
    NRCVoxelPT = createPass("NRCVoxelPT", {'params': PathTracerParams(useVBuffer=0, maxBounces=15, maxNonSpecularBounces=15)})
    g.addPass(NRCVoxelPT, "NRCVoxelPT")
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
    g.addEdge("NRCVoxelPT.result", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "NRCToneMapped.src")
    g.markOutput("NRCToneMapped.dst")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None

# Scene
m.loadScene('C:/Dependencies/ORCA/breakfast_room/breakfast_room.pyscene')
m.scene.renderSettings = SceneRenderSettings(useEnvLight=True, useAnalyticLights=True, useEmissiveLights=True, useVolumes=True)
m.scene.camera.position = float3(-4.633325,2.405962,7.047087)
m.scene.camera.target = float3(-3.948584,2.134528,6.370734)
m.scene.camera.up = float3(-0.000634,1.000000,0.000626)
m.scene.cameraSpeed = 1.0

# Window Configuration
m.resizeSwapChain(1920, 1080)
m.ui = True

# Clock Settings
m.clock.time = 0
m.clock.framerate = 0
# If framerate is not zero, you can use the frame property to set the start frame
# m.clock.frame = 0

# Frame Capture
m.frameCapture.outputDir = '.'
m.frameCapture.baseFilename = 'Mogwai'

