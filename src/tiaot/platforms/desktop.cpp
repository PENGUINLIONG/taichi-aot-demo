#include <cstdint>

#include <GLFW/glfw3.h>
#include "tiaot/framework.hpp"

int main(int argc, const char** argv) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  tiaot::AppConfig cfg {};
  cfg.user_data = nullptr;
  cfg.title = "Taichi AOT Demo";
  cfg.width = 512;
  cfg.height = 512;
  tiaot::configurate(argc, argv, &cfg);

  GLFWwindow* window = glfwCreateWindow(cfg.width, cfg.height, cfg.title, NULL, NULL);
  if (window == NULL) {
    TIAOT_ERROR("failed to create glfw window");
    glfwTerminate();
    return -1;
  }

  tiaot::AppData data {};
  tiaot::initialize(&data);
  while (!glfwWindowShouldClose(window) && !data.should_terminate) {
    tiaot::update(&data);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  tiaot::finalize(&data);

  return 0;
}
