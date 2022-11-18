#include <iostream>
#include <chrono>
#include "taichi/aot_demo/framework.hpp"
#include "gft/args.hpp"
#include "gft/util.hpp"
#include <android/log.h>
#include <android_native_app_glue.h>
#define VK_USE_PLATFORM_ANDROID_KHR 1
#include <vulkan/vulkan.h>

struct Config {
  TiArch arch = TI_ARCH_VULKAN;
  bool debug = false;
} CFG;

void initialize(const char* app_name, int argc, const char** argv) {
  namespace args = liong::args;
  std::string arch_lit = "vulkan";

  args::init_arg_parse(app_name, "One of the Taichi AOT demos.");
  args::reg_arg<args::StringParser>("", "--arch", arch_lit,
    "Arch of Taichi runtime.");
  args::reg_arg<args::SwitchParser>("", "--debug", CFG.debug,
    "Enable Vulkan validation layers.");

  args::parse_args(argc, argv);

  if (arch_lit == "vulkan") {
    CFG.arch = TI_ARCH_VULKAN;
  } else {
    throw std::runtime_error("unsupported arch");
  }
};

namespace {

// Assets are loaded via Android asset manager.
class AndroidAssetManager : public AssetManager {
  AAssetManager* asset_mgr_;

public:
  AndroidAssetManager(AAssetManager* asset_mgr) : asset_mgr_(asset_mgr) {}

  virtual bool load_file(const char* path, std::vector<uint8_t>& data) override final {
    data = liong::util::load_file(path);
    return data.size() != 0;
  }
  virtual bool load_text(const char* path, std::string& str) override final{
    str = liong::util::load_text(path);
    return str.size() != 0;
  }
};

} // namespace

std::unique_ptr<ti::aot_demo::AssetManager> create_asset_manager(AAssetManager* asset_mgr) {
  return std::unique_ptr<ti::aot_demo::AssetManager>(new AndroidAssetManager(asset_mgr));
}

namespace {

struct AndroidApp {
  std::unique_ptr<App> app = nullptr;
  bool is_active = false;
};

static void on_app_cmd_callback(struct android_app* state, int32_t cmd) {
  AndroidApp* app = (AndroidApp*)state->userData;
  switch (cmd) {
    case APP_CMD_INIT_WINDOW:
      std::unique_ptr<App> app2 = create_app();
      app2->initialize();
      ti::aot_demo::F = ti::aot_demo::Framework(app2->cfg(), CFG.arch, CFG.debug);
      F.renderer().set_surface_window(state->window);
      app->app = std::move(app2);
      break;
    case APP_CMD_TERM_WINDOW:
      app->app.reset();
      break;
    case APP_CMD_GAINED_FOCUS:
      app->is_active = true;
      break;
    case APP_CMD_LOST_FOCUS:
      app->is_active = false;
      break;
    default:
      break;
  }
}

} // namespace


void android_main(struct android_app* state) {
  std::unique_ptr<AndroidApp> app = std::make_unique<AndroidApp>();
  state->userData = app.get();
  state->onAppCmd = on_app_cmd_callback;

  while (true) {
    // Read all pending events.
    int events;
    struct android_poll_source* source;

    // If not animating, we will block forever waiting for events.
    // If animating, we loop until all events are read, then continue
    // to draw the next frame of animation.
    while ((ALooper_pollAll(app->is_active ? 0 : -1, nullptr, &events, (void**)&source)) >= 0) {
      if (source != nullptr) {
        source->process(state, source);
      }

      if (state->destroyRequested != 0) {
        app.reset();
        return;
      }
    }

    if (app->is_active) {
      if (!app->app->update()) {
        F.next_frame();
        return;
      }

      renderer.begin_render();
      app->app->render();
      renderer.end_render();

      renderer.present_to_surface();

      F.next_frame();
    }
  }
}
