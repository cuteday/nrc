#include "PassLibraryTemplate.h"


namespace
{
    const char kDesc[] = "Insert pass description here";    
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("RenderPassTemplate", kDesc, RenderPassTemplate::create);
}

RenderPassTemplate::SharedPtr RenderPassTemplate::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new RenderPassTemplate);
    return pPass;
}

std::string RenderPassTemplate::getDesc() { return kDesc; }

Dictionary RenderPassTemplate::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection RenderPassTemplate::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput("dst");
    //reflector.addInput("src");
    return reflector;
}

void RenderPassTemplate::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData["src"]->asTexture();
}

void RenderPassTemplate::renderUI(Gui::Widgets& widget)
{
}
