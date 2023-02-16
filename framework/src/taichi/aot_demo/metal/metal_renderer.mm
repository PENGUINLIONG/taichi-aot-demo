#include "taichi/aot_demo/metal/metal_renderer.hpp"
#include "taichi/aot_demo/metal/metal_graphics_task.hpp"

#ifndef __OBJC__
static_assert(false, "objective-c++ source");
#endif // __OBJC__

namespace ti {
namespace aot_demo {

RendererState::RendererState(const RendererConfig &config) {
  assert(config.client_arch == TI_ARCH_METAL);

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> command_queue = [device newCommandQueue];

  uint32_t width = config.framebuffer_width;
  uint32_t height = config.framebuffer_height;

  id<MTLTexture> color_attachment = nil;
  id<MTLTexture> depth_attachment = nil;
  @autoreleasepool {
    MTLTextureDescriptor *color_attachment_desc =
        [[MTLTextureDescriptor new] autorelease];
    color_attachment_desc.textureType = MTLTextureType2D;
    color_attachment_desc.width = config.framebuffer_width;
    color_attachment_desc.height = config.framebuffer_height;
    color_attachment_desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    color_attachment_desc.usage = MTLTextureUsageRenderTarget;

    color_attachment = [device newTextureWithDescriptor:color_attachment_desc];

    MTLTextureDescriptor *depth_attachment_desc =
        [[MTLTextureDescriptor new] autorelease];
    depth_attachment_desc.textureType = MTLTextureType2D;
    depth_attachment_desc.width = width_;
    depth_attachment_desc.height = height_;
    depth_attachment_desc.pixelFormat = MTLPixelFormatDepth32Float;
    depth_attachment_desc.usage = MTLTextureUsageRenderTarget;

    depth_attachment = [device newTextureWithDescriptor:depth_attachment_desc];
  }

  device_ = device;
  command_queue_ = command_queue;

  color_attachment_ = color_attachment;
  depth_attachment_ = depth_attachment;
  width_ = width;
  height_ = height;

  metal_layer_ = nil;
  metal_drawable_ = nil;

  frame_command_buffer_ = nil;
  render_command_encoder_ = nil;
  present_command_buffer_ = nil;
}
RendererState::~RendererState() {
  assert(frame_command_buffer_ == nil);
  assert(present_command_buffer_ == nil);

  [color_attachment_ release];
  [depth_attachment_ release];

  [command_queue_ release];
  [device_ release];
}

void RendererState::set_surface_window(const CAMetalLayer *layer) {
  metal_layer_ = layer;
}


const TiMetalMemoryInteropInfo &
RendererState::export_ti_memory(const ti::Runtime &runtime,
                                const ShadowBuffer &shadow_buffer) {
  TiMemory memory = shadow_buffer.renderer_memory();

  auto it = ti_memory_interops_.find(memory);
  if (it == ti_memory_interops_.end()) {
    TiMetalMemoryInteropInfo mmii{};
    ti_export_metal_memory(runtime, memory, &mmii);
    check_taichi_error();

    it = ti_memory_interops_.emplace(std::make_pair(memory, std::move(mmii))).first;
  }
  return it->second;
}

const TiMetalImageInteropInfo &
RendererState::export_ti_image(const ti::Runtime &runtime,
                               const ShadowTexture &shadow_texture) {
  TiImage image = shadow_texture.renderer_image();

  auto it = ti_image_interops_.find(image);
  if (it == ti_image_interops_.end()) {
    TiMetalImageInteropInfo miii {};
    ti_export_metal_image(runtime, image, &miii);
    check_taichi_error();

    it = ti_image_interops_.emplace(std::make_pair(image, std::move(miii))).first;
  }
  return it->second;
}



Renderer::Renderer(const RendererConfig &config)
    : state_(std::make_unique<RendererState>(config)) {}
Renderer::~Renderer() { destroy(); }

void Renderer::destroy() {
  staging_buffers_.clear();
  graphics_tasks_.clear();

  rect_vertex_buffer_.destroy();
  rect_texcoord_buffer_.destroy();

  runtime_.destroy();

  state_.reset();
}

void Renderer::begin_render() {
  assert(state_->frame_command_buffer_ == nil);

  id<MTLCommandBuffer> command_buffer = nil;
  id<MTLRenderCommandEncoder> render_command_encoder = nil;
  @autoreleasepool {
    MTLRenderPassDescriptor *desc =
        [MTLRenderPassDescriptor renderPassDescriptor];
    desc.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    desc.colorAttachments[0].loadAction  = MTLLoadActionClear;
    desc.colorAttachments[0].storeAction = MTLStoreActionStore;
    desc.colorAttachments[0].texture = state_->color_attachment_;
    desc.depthAttachment.clearDepth = 1.0f;
    desc.depthAttachment.loadAction = MTLLoadActionClear;
    desc.depthAttachment.storeAction = MTLStoreActionStore;
    desc.depthAttachment.texture = state_->depth_attachment_;

    command_buffer = [[state_->command_queue_ commandBuffer] retain];
    render_command_encoder =
        [[command_buffer renderCommandEncoderWithDescriptor:desc] retain];
  }

  state_->frame_command_buffer_ = command_buffer;
  state_->render_command_encoder_ = render_command_encoder;
}
void Renderer::end_render() {
  assert(state_->frame_command_buffer_ != nil);
  assert(state_->render_command_encoder_ != nil);

  [state_->render_command_encoder_ endEncoding];
  [state_->render_command_encoder_ release];
  [state_->frame_command_buffer_ commit];
}
void Renderer::enqueue_graphics_task(
    const std::shared_ptr<GraphicsTask> &task) {

  const GraphicsTaskConfig &config = task->state()->config_;
  bool is_indexed = config.index_buffer != nullptr;

  MTLPrimitiveType primitive_type;
  switch (config.primitive_topology) {
  case L_PRIMITIVE_TOPOLOGY_POINT:
    primitive_type = MTLPrimitiveTypePoint;
    break;
  case L_PRIMITIVE_TOPOLOGY_LINE:
    primitive_type = MTLPrimitiveTypeLine;
    break;
  case L_PRIMITIVE_TOPOLOGY_TRIANGLE:
    primitive_type = MTLPrimitiveTypeTriangle;
    break;
  default:
    assert(false);
  }

  [state_->render_command_encoder_ setVertexBuffer: offset:(NSUInteger) atIndex:(NSUInteger)]
  [state_->render_command_encoder_ drawPrimitives:primitive_type vertexStart:0 vertexCount:config.vertex_count];
}

void Renderer::present_to_surface() {
  assert(state_->metal_layer_ != nil);
  @autoreleasepool {
    id<CAMetalDrawable> drawable = [state_->metal_layer_ nextDrawable];
    id<MTLCommandBuffer> command_buffer =
        [state_->command_queue_ commandBuffer];

    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    [encoder endEncoding];

    [command_buffer presentDrawable:drawable];
    [command_buffer commit];

    state_->present_command_buffer_ = [command_buffer retain];
  }
}
ti::NdArray<uint8_t> Renderer::present_to_ndarray() {
  assert(state_->present_command_buffer_ == nil);

  ti::NdArray<uint8_t> dst = runtime_.allocate_ndarray<uint8_t>(
      {state_->width_, state_->height_}, {4}, true);

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer =
        [state_->command_queue_ commandBuffer];

    [command_buffer commit];

    state_->present_command_buffer_ = [command_buffer retain];
  }

  return dst;
}

void Renderer::next_frame() {
  [state_->frame_command_buffer_ waitUntilCompleted];
  [state_->frame_command_buffer_ release];
  state_->frame_command_buffer_ = nil;

  [state_->present_command_buffer_ waitUntilCompleted];
  [state_->present_command_buffer_ release];
  state_->present_command_buffer_ = nil;
}

}  // namespace aot_demo
}  // namespace ti
