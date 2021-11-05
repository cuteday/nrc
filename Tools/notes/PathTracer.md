# PathTracer

### Methods



### Debugging

#### Embedded Shader Program

`PixelDebug`和`PixelStats`就是两个典型的shader program。使用它们的shader需要`__exported import`这两个slang shader，在host code端create它们，并在`beginFrame`和`endFrame`中也调用它们的相应方法，以使它们可以向shader绑定资源。

#### Pixel Stats

`PixelStats.slang`使用Texture记录路径长度、采样次数等信息，在host code端使用`ParallelReduction`将Texture2D求和。在获取信息时使用了`CpuFence`进行同步(保证ParallelReduction已经完成)。在shader中使用`log-`对相应统计信息进行写入。

#### Pixel Debug

​	`PixelDebug.slang`绑定了一个固定长度的StructuredBuffer，在shader中可以使用`print(msg, val)`指令写入debug message，具体可见shader文件。使用pixel stats和pixel debug功能时都需要在pixel shader中通过`logSetPixel`和`printSetPixel`指定当前像素的坐标，这存储在`static int`中。

### Shader-CPU Communications

#### Mapping to structured buffer

`Buffer`类可以创建可被CPU读写的buffer，通过`Buffer::map`可以将buffer映射至内存中。除了原子类型，还可以使用自定义的数据结构`Buffer::createStructured`，在创建之时就要绑定到shader中的RWStructuredBuffer上，它就像是一个固定长度的数组一样被索引。

目前来看，structuredBuffer可以绑定到shader上，buffer虽然同样位于GPU中，但不能被绑定到shader上；而一般通过host code使用其他算子读写buffer，比如`ParallelReduction`。[开销尚待考察]

StructuredBuffer还会自带一个UAV Counter buffer，在shader中可以使用`IncrementCounter`等方法对其值进行修改，在host code中可以获取该buffer并`map`映射到内存中从而读取counter的值。

#### Ansynchronous routines

一些操作可以异步执行，比如说`copyBufferRegion`或者`executeParallelReduction`等。这些操作会被提交到command list中，我们使用`RenderContext::Flush`提交这些commend，并可以指定是否阻塞直到任务完成。一般我们与`GpuFence`联合使用，可在这些命令执行完成时进行同步。

#### CUDA Interop

我们目前还在考察，使用CUDA interop能否将shader buffer中的内容在避免经过CPU的情况下直接送入CUDA，现在看来好像是可行的。在官方的`Cudainterop` sample中描述了将Texture映射到`CudaExternalMemory`中的流程。

