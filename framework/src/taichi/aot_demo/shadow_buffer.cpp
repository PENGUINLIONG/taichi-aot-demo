#include "taichi/aot_demo/shadow_buffer.hpp"
#include "taichi/aot_demo/renderer.hpp"

namespace ti {
namespace aot_demo {

ShadowBuffer::ShadowBuffer(const std::shared_ptr<Renderer> &renderer,
               const ti::Memory &client_memory,
               ShadowBufferUsage usage)
    : renderer_(renderer) {
  TiMemoryAllocateInfo mai {};
  mai.size = client_memory.size();
  mai.export_sharing = TI_TRUE;
  switch (usage) {
    case ShadowBufferUsage::VertexBuffer:
      mai.usage = TI_MEMORY_USAGE_STORAGE_BIT | TI_MEMORY_USAGE_VERTEX_BIT;
      break;
    case ShadowBufferUsage::IndexBuffer:
      mai.usage = TI_MEMORY_USAGE_STORAGE_BIT | TI_MEMORY_USAGE_INDEX_BIT;
      break;
    case ShadowBufferUsage::StorageBuffer:
      mai.usage = TI_MEMORY_USAGE_STORAGE_BIT;
      break;
  }
  ti::Memory memory = renderer->runtime_.allocate_memory(mai);

  usage_ = usage;
  memory_ = std::move(memory);
  client_memory_ = ti::Memory(renderer_->client_runtime_, client_memory,
                              client_memory.size(), false);
}

ShadowBuffer::~ShadowBuffer() {
  memory_.destroy();
}


void ShadowBuffer::copy_from_vulkan_() {
  const ti::Runtime &client_runtime = renderer_->client_runtime();

  client_memory_.copy_to(memory_);
}
void ShadowBuffer::copy_from_cpu_() {
#if TI_WITH_CPU
  const ti::Runtime &client_runtime = renderer_->client_runtime();

  TiCpuMemoryInteropInfo cmii{};
  ti_export_cpu_memory(client_runtime, client_memory_.memory(), &cmii);

  ti::Memory staging_buffer = renderer_.allocate_staging_buffer(memory_.size());
  staging_buffer.write(cmii.ptr, cmii.size);
  staging_buffer.copy_to(memory_);
#endif // TI_WITH_CPU
}

void ShadowBuffer::update() {
  if (renderer_->renderer_runtime().arch() ==
      renderer_->client_runtime().arch()) {
    copy_from_same_arch_();
    return;
  }
  switch (renderer_->client_runtime_.arch()) {
    case TI_ARCH_VULKAN:
      copy_from_vulkan_();
      break;
    case TI_ARCH_METAL:
      copy_from_metal_();
      break;
    case TI_ARCH_X64:
    case TI_ARCH_ARM64:
      copy_from_cpu_();
      break;
    case TI_ARCH_CUDA:
      copy_from_cuda_();
      break;
    case TI_ARCH_OPENGL:
      copy_from_opengl_();
      break;
    default:
      assert(false);
  }
}


} // namespace aot_demo
} // namespace ti
