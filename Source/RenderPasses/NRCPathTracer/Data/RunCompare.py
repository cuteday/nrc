from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    loadRenderPassLibrary('BSDFViewer.dll')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('Antialiasing.dll')
    loadRenderPassLibrary('DepthPass.dll')
    loadRenderPassLibrary('DebugPasses.dll')
    loadRenderPassLibrary('BlitPass.dll')
    loadRenderPassLibrary('CSM.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('ErrorMeasurePass.dll')
    loadRenderPassLibrary('FLIPPass.dll')
    loadRenderPassLibrary('TemporalDelayPass.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    loadRenderPassLibrary('ForwardLightingPass.dll')
    loadRenderPassLibrary('ImageLoader.dll')
    loadRenderPassLibrary('MegakernelPathTracer.dll')
    loadRenderPassLibrary('TestPasses.dll')
    loadRenderPassLibrary('MinimalPathTracer.dll')
    loadRenderPassLibrary('RTGeometryPass.dll')
    loadRenderPassLibrary('NRCPathTracer.dll')
    loadRenderPassLibrary('OptixDenoiser.dll')
    loadRenderPassLibrary('PassLibraryTemplate.dll')
    loadRenderPassLibrary('RTAO.dll')
    loadRenderPassLibrary('RTLightingPass.dll')
    loadRenderPassLibrary('SceneDebugger.dll')
    loadRenderPassLibrary('SimplePostFX.dll')
    loadRenderPassLibrary('SSAO.dll')
    loadRenderPassLibrary('SkyBox.dll')
    loadRenderPassLibrary('SVGFPass.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    loadRenderPassLibrary('Utils.dll')
    loadRenderPassLibrary('WhittedRayTracer.dll')
    ImageLoader = createPass('ImageLoader', {'filename': '', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoader, 'ImageLoader')
    SideBySidePass = createPass('SideBySidePass', {'splitLocation': -1.0, 'showTextLabels': False, 'leftLabel': 'Left side', 'rightLabel': 'Right side'})
    g.addPass(SideBySidePass, 'SideBySidePass')
    SplitScreenPass = createPass('SplitScreenPass', {'splitLocation': 0.0, 'showTextLabels': False, 'leftLabel': 'Left side', 'rightLabel': 'Right side'})
    g.addPass(SplitScreenPass, 'SplitScreenPass')
    ImageLoader_ = createPass('ImageLoader', {'filename': '', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoader_, 'ImageLoader_')
    g.addEdge('ImageLoader.dst', 'SplitScreenPass.leftInput')
    g.addEdge('ImageLoader_.dst', 'SplitScreenPass.rightInput')
    g.addEdge('ImageLoader_.dst', 'SideBySidePass.rightInput')
    g.addEdge('ImageLoader.dst', 'SideBySidePass.leftInput')
    g.markOutput('SplitScreenPass.output')
    g.markOutput('SideBySidePass.output')
    return g

DefaultRenderGraph = render_graph_DefaultRenderGraph()
try: m.addGraph(DefaultRenderGraph)
except NameError: None
