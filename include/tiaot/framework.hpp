// Logging infrastructure.
// @PENGUINLIONG
#pragma once
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <vector>
#include <string>

#define TIAOT_LOG(level, f, ...) \
    ::std::fprintf(stderr, "[" #level "] (" __FILE__ ") " f "\n",  __VA_ARGS__)

#define TIAOT_DEBUG(...) TIAOT_LOG(INFO, __VA_ARGS__)
#define TIAOT_INFO(...) TIAOT_LOG(INFO, __VA_ARGS__)
#define TIAOT_WARN(...) TIAOT_LOG(WARN, __VA_ARGS__)
#define TIAOT_ERROR(...) TIAOT_LOG(ERROR, __VA_ARGS__)

namespace tiaot {

struct AppConfig {
  // User provided application data. This is optionally assigned by the demo
  // implementations to pass persistent states between frames.
  void* user_data;
  // Window title. Should point to a string literal. DO NOT assign with a
  // `std::string`.
  const char* title;
  // Desired render target width.
  uint32_t width;
  // Desired render target height.
  uint32_t height;
};

extern void initialize(int argc, const char** argv, AppConfig* cfg);

struct AppData {
  // This has the same value as `AppConfig::user_data`.
  void* user_data;
  // Time in seconds since the application starts. This is assigned by the
  // application shell before `tick`ing a frame.
  double t;
  // Time elapsed since the last frame. This is assigned by the application
  // shell before `tick`ing a frame.
  double dt;
  // Whether the application should terminate.
  bool should_terminate;
};

extern void update(AppData* data);
extern void finalize(AppData* data);

template<typename T>
inline std::vector<T> load_bin(const std::string& path) {
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::ate);
  if (!f.is_open()) {
    TIAOT_ERROR("cannot load binary file: %s", path.c_str());
    assert(false);
  }
  size_t size = f.tellg();
  f.seekg(std::ios::beg);
  std::vector<T> out(size / sizeof(T));
  f.read(out.data(), out.size());
  return out;
}

} // namespace tiaot
