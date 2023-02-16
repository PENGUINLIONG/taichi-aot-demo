#pragma once
#include "taichi/aot_demo/graphics_task.hpp"
#include "taichi/aot_demo/metal/metal_renderer.hpp"

namespace ti {
namespace aot_demo {

struct GraphicsTaskState {
  std::shared_ptr<Renderer> renderer_;
  GraphicsTaskConfig config_;

  id<MTLRenderPipelineState> render_pipeline_state_;
  id<MTLBuffer> uniform_buffer_;

  GraphicsTaskState(const std::shared_ptr<Renderer> &renderer,
                    const GraphicsTaskConfig &config);
  ~GraphicsTaskState();
};

}  // namespace aot_demo
}  // namespace ti
