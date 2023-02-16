#include "taichi/aot_demo/graphics_task.hpp"
#include "taichi/aot_demo/metal/metal_renderer.hpp"

namespace ti {
namespace aot_demo {

// Implemented in spirv_cross.cpp
extern std::string spv2msl(const std::vector<uint32_t> &spv);

// Implemented in glslang.cpp
std::vector<uint32_t> vert2spv(const std::string& vert);
std::vector<uint32_t> frag2spv(const std::string &frag);

id<MTLFunction> create_function(id<MTLDevice> device, const std::string &glsl) {
  std::string msl = spv2msl(vert2spv(glsl));

  id<MTLLibrary> mtl_library = nil;
  {
    NSError *err = nil;
    NSString *msl_ns = [[NSString alloc] initWithUTF8String:msl.c_str()];
    mtl_library = [device newLibraryWithSource:msl_ns options:nil error:&err];

    assert(mtl_library != nil);
    assert(err == nil);
  }

  id<MTLFunction> mtl_function = nil;
  {
    NSString *entry_name_ns = [[NSString alloc] initWithUTF8String:"main0"];
    mtl_function = [mtl_library newFunctionWithName:entry_name_ns];

    assert(mtl_function != nil);
  }

  [mtl_library release];
  return mtl_function;
}

GraphicsTaskState::GraphicsTaskState(const std::shared_ptr<Renderer> &renderer,
                                     const GraphicsTaskConfig &config)
    : renderer_(renderer), config_(config) {
  assert(renderer->is_valid());

  RendererState *renderer_state = renderer->state();

  id<MTLDevice> device = renderer_state->device_;

  id<MTLRenderPipelineState> render_pipeline_state = nil;
  @autoreleasepool {
    MTLRenderPipelineDescriptor *desc =
        [[MTLRenderPipelineDescriptor new] autorelease];
    desc.vertexFunction =
        [create_function(device, config.vertex_shader_glsl) autorelease];
    desc.fragmentFunction =
        [create_function(device, config.fragment_shader_glsl) autorelease];
    MTLVertexDescriptor *vert_desc = [[MTLVertexDescriptor new] autorelease];
    vert_desc.attributes[0].bufferIndex = 0;
    switch (config.vertex_component_count) {
    case 1:
      vert_desc.attributes[0].format = MTLVertexFormatFloat;
      break;
    case 2:
      vert_desc.attributes[0].format = MTLVertexFormatFloat2;
      break;
    case 3:
      vert_desc.attributes[0].format = MTLVertexFormatFloat3;
      break;
    case 4:
      vert_desc.attributes[0].format = MTLVertexFormatFloat4;
      break;
    default:
      assert(false);
    }
    desc.vertexDescriptor = vert_desc;

    NSError *err = nil;
    render_pipeline_state =
        [device newRenderPipelineStateWithDescriptor:desc error:&err];

    assert(render_pipeline_state != nil);
    assert(err == nil);
  }

  id<MTLBuffer> uniform_buffer = nil;
  {
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                                 MTLResourceStorageModeShared |
                                 MTLResourceHazardTrackingModeDefault;
    uniform_buffer = [device newBufferWithBytes:config.uniform_buffer_data
                                         length:config.uniform_buffer_size
                                        options:options];
  }

  render_pipeline_state_ = render_pipeline_state;
  uniform_buffer_ = uniform_buffer;
}
GraphicsTaskState::~GraphicsTaskState() {
  [uniform_buffer_ release];
  [render_pipeline_state_ release];
}

GraphicsTask::GraphicsTask(const std::shared_ptr<Renderer> &renderer,
                           const GraphicsTaskConfig &config)
    : state_(std::make_shared<GraphicsTaskState>(renderer, config)) {
}

}  // namespace aot_demo
}  // namespace ti
