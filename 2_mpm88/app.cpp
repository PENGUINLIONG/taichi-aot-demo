#include <memory>
#include <thread>
#include <chrono>
#include <iostream>
#include "glm/glm.hpp"
#include "taichi/aot_demo/draws/draw_points.hpp"
#include "taichi/aot_demo/framework.hpp"
#include "taichi/aot_demo/shadow_buffer.hpp"

using namespace ti::aot_demo;

static std::string get_aot_file_dir(TiArch arch) {
    switch(arch) {
        case TI_ARCH_VULKAN: {
            return "2_mpm88/assets/mpm88_vulkan";
        }
        case TI_ARCH_X64: {
            return "2_mpm88/assets/mpm88_x64";
        }
        case TI_ARCH_CUDA: {
            return "2_mpm88/assets/mpm88_cuda";
        }
        case TI_ARCH_OPENGL: {
            return "2_mpm88/assets/mpm88_opengl";
        }
        default: {
            throw std::runtime_error("Unrecognized arch");
        }
    }
}

struct App2_mpm88 : public App {
  static const uint32_t NPARTICLE = 8192 * 2;
  static const uint32_t GRID_SIZE = 128;

  ti::AotModule module_;

  ti::ComputeGraph g_init_;
  ti::ComputeGraph g_update_;

  ti::NdArray<float> x_;
  ti::NdArray<float> v_;
  ti::NdArray<float> pos_;
  ti::NdArray<float> C_;
  ti::NdArray<float> J_;
  ti::NdArray<float> grid_v_;
  ti::NdArray<float> grid_m_;

  std::unique_ptr<GraphicsTask> draw_points;

  virtual AppConfig cfg() const override final {
    AppConfig out {};
    out.app_name = "2_mpm88";
    out.framebuffer_width = 256;
    out.framebuffer_height = 256;
    out.supported_archs = {
      TI_ARCH_VULKAN,
      TI_ARCH_CUDA,
      TI_ARCH_X64,
    };
    return out;
  }

  virtual void initialize() override final{
    Renderer &renderer = F_->renderer();
    ti::Runtime &runtime = F_->runtime();

    // 2. Load AOT module
#ifdef TI_AOT_DEMO_WITH_ANDROID_APP
    std::vector<uint8_t> tcm;
    F_->asset_mgr().load_file("E2_mpm88.tcm", tcm);
    module_ = runtime.create_aot_module(tcm);
#else
    auto aot_file_path = get_aot_file_dir(runtime.arch());
    module_ = runtime.load_aot_module(aot_file_path);
#endif

    g_init_ = module_.get_compute_graph("init");
    g_update_ = module_.get_compute_graph("update");

    x_ = runtime.allocate_ndarray<float>({NPARTICLE}, {2});
    v_ = runtime.allocate_ndarray<float>({NPARTICLE}, {2});
    pos_ = runtime.allocate_ndarray<float>({NPARTICLE}, {3});
    C_ = runtime.allocate_ndarray<float>({NPARTICLE}, {2, 2});
    J_ = runtime.allocate_ndarray<float>({NPARTICLE}, {});
    grid_v_ = runtime.allocate_ndarray<float>({GRID_SIZE, GRID_SIZE}, {2});
    grid_m_ = runtime.allocate_ndarray<float>({GRID_SIZE, GRID_SIZE}, {});

    draw_points = renderer.draw_points(x_)
      .point_size(3.0f)
      .color(glm::vec3(0,0,1))
      .build();

    g_init_["x"] = x_;
    g_init_["v"] = v_;
    g_init_["J"] = J_;
    g_init_.launch();

    g_update_["x"] = x_;
    g_update_["v"] = v_;
    g_update_["pos"] = pos_;
    g_update_["C"] = C_;
    g_update_["J"] = J_;
    g_update_["grid_v"] = grid_v_;
    g_update_["grid_m"] = grid_m_;

    renderer.set_framebuffer_size(256, 256);

    std::cout << "initialized!" << std::endl;
  }
  virtual bool update() override final {
    g_update_.launch();

    std::cout << "stepped! (fps=" << F_->fps() << ")" << std::endl;
    return true;
  }
  virtual void render() override final {
    Renderer& renderer = F_->renderer();
    renderer.enqueue_graphics_task(*draw_points);
  }
};

std::unique_ptr<App> create_app() {
  return std::make_unique<App2_mpm88>();
}
