#pragma once
#include "taichi/aot_demo/renderer.hpp"

#ifndef TI_AOT_DEMO_RENDER_WITH_VULKAN
static_assert(TI_AOT_DEMO_RENDER_WITH_METAL,
              "metal renderer headers are only accessible when you are using a "
              "metal renderer");
#endif  // TI_AOT_DEMO_RENDER_WITH_VULKAN

namespace ti {
namespace aot_demo {

struct RendererState {
  id<MTLDevice> device_;
  id<MTLCommandQueue> command_queue_;

  id<MTLTexture> color_attachment_;
  id<MTLTexture> depth_attachment_;
  uint32_t width_;
  uint32_t height_;

  const CAMetalLayer *metal_layer_;
  id<CAMetalDrawable> metal_drawable_;

  id<MTLCommandBuffer> frame_command_buffer_;
  id<MTLRenderCommandEncoder> render_command_encoder_;
  id<MTLCommandBuffer> present_command_buffer_;

  RendererState(const RendererConfig &config);
  ~RendererState();

  void set_surface_window(const CAMetalLayer *layer);

  std::map<TiMemory, TiMetalMemoryInteropInfo> ti_memory_interops_;
  std::map<TiImage, TiMetalImageInteropInfo> ti_image_interops_;
  const TiMetalMemoryInteropInfo &export_ti_memory(
      const ti::Runtime &runtime,
      const ShadowBuffer &shadow_buffer);
  const TiMetalImageInteropInfo &export_ti_image(
      const ti::Runtime &runtime,
      const ShadowTexture &shadow_texture);
};

}  // namespace aot_demo
}  // namespace ti
