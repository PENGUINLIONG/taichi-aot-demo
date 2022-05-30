#pragma once

#define TI_WITH_VULKAN 1
#include "taichi/taichi.h"
#include "glm/glm.hpp"

#include <memory>
#include <vector>

#include "box_color_data.h"
#include "mesh_data.h"

constexpr float DT = 7.5e-3;
constexpr int NUM_SUBSTEPS = 2;
constexpr int CG_ITERS = 8;
constexpr float ASPECT_RATIO = 2.0f;

void load_data(TiDevice device,
               TiDeviceAllocation devalloc, const void* data,
               size_t size) {
  void* mapped = tiMapDeviceAllocation(device, devalloc);
  std::memcpy(mapped, data, size);
  tiUnmapDeviceAllocation(device, devalloc);
}

struct NdArray {
  
};


struct ColorVertex {
  glm::vec3 pos;
  glm::vec3 color;
};

void build_wall(int face, std::vector<ColorVertex>& vertices,
                std::vector<int>& indices, glm::vec3 axis_x, glm::vec3 axis_y,
                glm::vec3 base) {
  int base_vertex = int(vertices.size());

  for (int j = 0; j < 32; j++) {
    for (int i = 0; i < 32; i++) {
      glm::vec3 pos = base + axis_x * ((float(i) / 31.0f) * 2.0f - 1.0f) +
                      axis_y * ((float(j) / 31.0f) * 2.0f - 1.0f);
      pos.y *= ASPECT_RATIO;
      vertices.push_back(ColorVertex{pos, box_color_data[face][i + j * 32]});
    }
  }

  for (int j = 0; j < 31; j++) {
    for (int i = 0; i < 31; i++) {
      int i00 = base_vertex + (i + j * 32);
      int i01 = base_vertex + (i + (j + 1) * 32);
      int i10 = base_vertex + ((i + 1) + j * 32);
      int i11 = base_vertex + ((i + 1) + (j + 1) * 32);

      indices.push_back(i00);
      indices.push_back(i01);
      indices.push_back(i10);

      indices.push_back(i01);
      indices.push_back(i10);
      indices.push_back(i11);
    }
  }
}

#if false
class FemApp {
 public:
  void run_init(int width, int height, std::string path_prefix,
                taichi::ui::TaichiWindow* window) {
    using namespace taichi::lang;
    width_ = width;
    height_ = height;

#ifdef ANDROID
    const std::vector<std::string> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
#else
    std::vector<std::string> extensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };

    uint32_t glfw_ext_count = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    for (int i = 0; i < glfw_ext_count; ++i) {
      extensions.push_back(glfw_extensions[i]);
    }
#endif  // ANDROID
    // Create a Vulkan Device
    taichi::lang::vulkan::VulkanDeviceCreator::Params evd_params;
    evd_params.api_version = VK_API_VERSION_1_2;
    evd_params.additional_instance_extensions = extensions;
    evd_params.additional_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    evd_params.is_for_ui = false;
    evd_params.surface_creator = nullptr;
    // (penguinliong) vulkan context should be created outside.

    embedded_device_ =
        std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

    device_ = static_cast<taichi::lang::vulkan::VulkanDevice*>(
        embedded_device_->device());

    {
      taichi::lang::SurfaceConfig config;
      config.vsync = true;
      config.window_handle = window;
      config.width = width_;
      config.height = height_;
      surface_ = device_->create_surface(config);
    }

    {
      taichi::lang::ImageParams params;
      params.dimension = ImageDimension::d2D;
      params.format = BufferFormat::depth32f;
      params.initial_layout = ImageLayout::undefined;
      params.x = width_;
      params.y = height_;
      params.export_sharing = false;

      depth_allocation_ = device_->create_image(params);
    }

    // Initialize our Vulkan Program pipeline
    host_result_buffer_.resize(taichi_result_buffer_entries);
    taichi::lang::vulkan::VkRuntime::Params params;
    params.host_result_buffer = host_result_buffer_.data();
    params.device = embedded_device_->device();
    vulkan_runtime_ =
        std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

    std::string shader_source = path_prefix + "/shaders/aot/implicit_fem";
    taichi::lang::vulkan::AotModuleParams aot_params{shader_source,
                                                     vulkan_runtime_.get()};
    module_ = taichi::lang::aot::Module::load(taichi::Arch::vulkan, aot_params);
    auto root_size = module_->get_root_size();
    // printf("root buffer size=%ld\n", root_size);
    vulkan_runtime_->add_root_buffer(root_size);

    loaded_kernels_.get_force_kernel = module_->get_kernel("get_force");
    loaded_kernels_.init_kernel = module_->get_kernel("init");
    loaded_kernels_.floor_bound_kernel = module_->get_kernel("floor_bound");
    loaded_kernels_.get_matrix_kernel = module_->get_kernel("get_matrix");
    loaded_kernels_.matmul_edge_kernel = module_->get_kernel("matmul_edge");
    loaded_kernels_.add_kernel = module_->get_kernel("add");
    loaded_kernels_.add_scalar_ndarray_kernel =
        module_->get_kernel("add_scalar_ndarray");
    loaded_kernels_.dot2scalar_kernel = module_->get_kernel("dot2scalar");
    loaded_kernels_.get_b_kernel = module_->get_kernel("get_b");
    loaded_kernels_.ndarray_to_ndarray_kernel =
        module_->get_kernel("ndarray_to_ndarray");
    loaded_kernels_.fill_ndarray_kernel = module_->get_kernel("fill_ndarray");
    loaded_kernels_.clear_field_kernel = module_->get_kernel("clear_field");
    loaded_kernels_.init_r_2_kernel = module_->get_kernel("init_r_2");
    loaded_kernels_.update_alpha_kernel = module_->get_kernel("update_alpha");
    loaded_kernels_.update_beta_r_2_kernel =
        module_->get_kernel("update_beta_r_2");

    using DeviceAllocation = taichi::lang::DeviceAllocation*;
    using AllocParams = taichi::lang::Device::AllocParams;
    struct ContextSetArgumentDeviceAllocationParams {
      uint32_t shape_dims;
      uint32_t* shape;
      uint32_t elem_shape_dims;
      uint32_t* elem_shape;
    };
    struct ContextSetArgumentScalarParams {
      int64_t int_scalar;
      uint64_t uint_scalar;
      float float_scalar;
    };
    struct ContextSetArgumentParams {
      uint32_t index;
      union {
        ContextSetArgumentDeviceAllocationParams devalloc_params;
        ContextSetArgumentScalarParams scalar_params;
      };
    };

    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams alloc_params;
    alloc_params.host_write = true;
    // x
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    alloc_params.usage =
        taichi::lang::AllocUsage::Vertex | taichi::lang::AllocUsage::Storage;
    devalloc_x_ = device_->allocate_memory(alloc_params);
    alloc_params.usage = taichi::lang::AllocUsage::Storage;
    // v
    devalloc_v_ = device_->allocate_memory(alloc_params);
    // f
    devalloc_f_ = device_->allocate_memory(alloc_params);
    // mul_ans
    devalloc_mul_ans_ = device_->allocate_memory(alloc_params);
    // c2e
    alloc_params.size = N_CELLS * 6 * sizeof(int);
    devalloc_c2e_ = device_->allocate_memory(alloc_params);
    // b
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    devalloc_b_ = device_->allocate_memory(alloc_params);
    // r0
    devalloc_r0_ = device_->allocate_memory(alloc_params);
    // p0
    devalloc_p0_ = device_->allocate_memory(alloc_params);
    // indices
    alloc_params.size = N_FACES * 3 * sizeof(int);
    alloc_params.usage = taichi::lang::AllocUsage::Index;
    devalloc_indices_ = device_->allocate_memory(alloc_params);
    alloc_params.usage = taichi::lang::AllocUsage::Storage;
    // vertices
    alloc_params.size = N_CELLS * 4 * sizeof(int);
    devalloc_vertices_ = device_->allocate_memory(alloc_params);
    // edges
    alloc_params.size = N_EDGES * 2 * sizeof(int);
    devalloc_edges_ = device_->allocate_memory(alloc_params);
    // ox
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    devalloc_ox_ = device_->allocate_memory(alloc_params);

    alloc_params.size = sizeof(float);
    devalloc_alpha_scalar_ = device_->allocate_memory(alloc_params);
    devalloc_beta_scalar_ = device_->allocate_memory(alloc_params);

    load_data(vulkan_runtime_.get(), devalloc_indices_, indices_data,
              sizeof(indices_data));
    load_data(vulkan_runtime_.get(), devalloc_c2e_, c2e_data, sizeof(c2e_data));
    load_data(vulkan_runtime_.get(), devalloc_vertices_, vertices_data,
              sizeof(vertices_data));
    load_data(vulkan_runtime_.get(), devalloc_ox_, ox_data, sizeof(ox_data));
    load_data(vulkan_runtime_.get(), devalloc_edges_, edges_data,
              sizeof(edges_data));

    memset(&host_ctx_, 0, sizeof(taichi::lang::RuntimeContext));
    host_ctx_.result_buffer = host_result_buffer_.data();
    loaded_kernels_.clear_field_kernel->launch(&host_ctx_);

    host_ctx_.set_arg_devalloc(0, devalloc_x_, {N_VERTS}, {3, 1});
    host_ctx_.set_arg_devalloc(1, devalloc_v_, {N_VERTS}, {3, 1});
    host_ctx_.set_arg_devalloc(2, devalloc_f_, {N_VERTS}, {3, 1});
    host_ctx_.set_arg_devalloc(3, devalloc_ox_, {N_VERTS}, {3, 1});
    host_ctx_.set_arg_devalloc(4, devalloc_vertices_, {N_CELLS}, {4, 1});
    // init(x, v, f, ox, vertices)
    loaded_kernels_.init_kernel->launch(&host_ctx_);
    // get_matrix(c2e, vertices)
    host_ctx_.set_arg_devalloc(0, devalloc_c2e_, {N_CELLS}, {6, 1});
    host_ctx_.set_arg_devalloc(1, devalloc_vertices_, {N_CELLS}, {4, 1});
    loaded_kernels_.get_matrix_kernel->launch(&host_ctx_);
    vulkan_runtime_->synchronize();

    {
      build_wall(0, cornell_box_vertices_, cornell_box_indicies_,
                 glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0),
                 glm::vec3(-1.0, 0.0, 0.0));
      build_wall(1, cornell_box_vertices_, cornell_box_indicies_,
                 glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0),
                 glm::vec3(1.0, 0.0, 0.0));
      build_wall(2, cornell_box_vertices_, cornell_box_indicies_,
                 glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 0.0, 0.0),
                 glm::vec3(0.0, 1.0, 0.0));
      build_wall(3, cornell_box_vertices_, cornell_box_indicies_,
                 glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 0.0, 0.0),
                 glm::vec3(0.0, -1.0, 0.0));
      build_wall(4, cornell_box_vertices_, cornell_box_indicies_,
                 glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 0.0, 0.0),
                 glm::vec3(0.0, 0.0, -1.0));
    }

    {
      auto vert_code =
          taichi::ui::read_file(path_prefix + "/shaders/render/box.vert.spv");
      auto frag_code =
          taichi::ui::read_file(path_prefix + "/shaders/render/box.frag.spv");

      std::vector<PipelineSourceDesc> source(2);
      source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
                   frag_code.size(), PipelineStageType::fragment};
      source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
                   vert_code.size(), PipelineStageType::vertex};

      RasterParams raster_params;
      raster_params.prim_topology = TopologyType::Triangles;
      raster_params.polygon_mode = PolygonMode::Fill;
      raster_params.depth_test = true;
      raster_params.depth_write = true;

      std::vector<VertexInputBinding> vertex_inputs = {
          {/*binding=*/0, /*stride=*/6 * sizeof(float), /*instance=*/false}};
      std::vector<VertexInputAttribute> vertex_attribs;
      vertex_attribs.push_back({/*location=*/0, /*binding=*/0,
                                /*format=*/BufferFormat::rgb32f,
                                /*offset=*/0});
      vertex_attribs.push_back({/*location=*/1, /*binding=*/0,
                                /*format=*/BufferFormat::rgb32f,
                                /*offset=*/3 * sizeof(float)});

      render_box_pipeline_ = device_->create_raster_pipeline(
          source, raster_params, vertex_inputs, vertex_attribs);

      alloc_params = Device::AllocParams{};
      alloc_params.host_write = true;
      // x
      alloc_params.size = sizeof(ColorVertex) * cornell_box_vertices_.size();
      alloc_params.usage = taichi::lang::AllocUsage::Vertex;
      devalloc_box_verts_ = device_->allocate_memory(alloc_params);
      alloc_params.size = sizeof(int) * cornell_box_indicies_.size();
      alloc_params.usage = taichi::lang::AllocUsage::Index;
      devalloc_box_indices_ = device_->allocate_memory(alloc_params);
      load_data(vulkan_runtime_.get(), devalloc_box_verts_,
                cornell_box_vertices_.data(),
                sizeof(ColorVertex) * cornell_box_vertices_.size());
      load_data(vulkan_runtime_.get(), devalloc_box_indices_,
                cornell_box_indicies_.data(),
                sizeof(int) * cornell_box_indicies_.size());
    }
    {
      auto vert_code = taichi::ui::read_file(
          path_prefix + "/shaders/render/surface.vert.spv");
      auto frag_code = taichi::ui::read_file(
          path_prefix + "/shaders/render/surface.frag.spv");

      std::vector<PipelineSourceDesc> source(2);
      source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
                   frag_code.size(), PipelineStageType::fragment};
      source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
                   vert_code.size(), PipelineStageType::vertex};

      RasterParams raster_params;
      raster_params.prim_topology = TopologyType::Triangles;
      raster_params.depth_test = true;
      raster_params.depth_write = true;

      std::vector<VertexInputBinding> vertex_inputs = {
          {/*binding=*/0, /*stride=*/3 * sizeof(float), /*instance=*/false}};
      std::vector<VertexInputAttribute> vertex_attribs;
      vertex_attribs.push_back({/*location=*/0, /*binding=*/0,
                                /*format=*/BufferFormat::rgb32f,
                                /*offset=*/0});

      render_mesh_pipeline_ = device_->create_raster_pipeline(
          source, raster_params, vertex_inputs, vertex_attribs);
    }

    render_constants_ = device_->allocate_memory(
        {sizeof(RenderConstants), true, false, false, AllocUsage::Uniform});
  }

  void run_render_loop(float g_x = 0, float g_y = -9.8, float g_z = 0) {
    using namespace taichi::lang;
    for (int i = 0; i < NUM_SUBSTEPS; i++) {
      // get_force(x, f, vertices)
      host_ctx_.set_arg_devalloc(0, devalloc_x_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(1, devalloc_f_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(2, devalloc_vertices_, {N_CELLS}, {4, 1});
      host_ctx_.set_arg<float>(3, g_x);
      host_ctx_.set_arg<float>(4, g_y);
      host_ctx_.set_arg<float>(5, g_z);
      loaded_kernels_.get_force_kernel->launch(&host_ctx_);
      // get_b(v, b, f)
      host_ctx_.set_arg_devalloc(0, devalloc_v_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(1, devalloc_b_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(2, devalloc_f_, {N_VERTS}, {3, 1});
      loaded_kernels_.get_b_kernel->launch(&host_ctx_);

      // matmul_edge(mul_ans, v, edges)
      host_ctx_.set_arg_devalloc(0, devalloc_mul_ans_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(1, devalloc_v_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(2, devalloc_edges_, {N_EDGES}, {2, 1});
      loaded_kernels_.matmul_edge_kernel->launch(&host_ctx_);
      // add(r0, b, -1, mul_ans)
      host_ctx_.set_arg_devalloc(0, devalloc_r0_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(1, devalloc_b_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg<float>(2, -1.0f);
      host_ctx_.set_arg_devalloc(3, devalloc_mul_ans_, {N_VERTS}, {3, 1});
      loaded_kernels_.add_kernel->launch(&host_ctx_);
      // ndarray_to_ndarray(p0, r0)
      host_ctx_.set_arg_devalloc(0, devalloc_p0_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(1, devalloc_r0_, {N_VERTS}, {3, 1});
      loaded_kernels_.ndarray_to_ndarray_kernel->launch(&host_ctx_);
      // dot2scalar(r0, r0)
      host_ctx_.set_arg_devalloc(0, devalloc_r0_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(1, devalloc_r0_, {N_VERTS}, {3, 1});
      loaded_kernels_.dot2scalar_kernel->launch(&host_ctx_);
      // init_r_2()
      loaded_kernels_.init_r_2_kernel->launch(&host_ctx_);

      for (int i = 0; i < CG_ITERS; i++) {
        // matmul_edge(mul_ans, p0, edges);
        host_ctx_.set_arg_devalloc(0, devalloc_mul_ans_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg_devalloc(1, devalloc_p0_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg_devalloc(2, devalloc_edges_, {N_EDGES}, {2, 1});
        loaded_kernels_.matmul_edge_kernel->launch(&host_ctx_);
        // dot2scalar(p0, mul_ans)
        host_ctx_.set_arg_devalloc(0, devalloc_p0_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg_devalloc(1, devalloc_mul_ans_, {N_VERTS}, {3, 1});
        loaded_kernels_.dot2scalar_kernel->launch(&host_ctx_);
        host_ctx_.set_arg_devalloc(0, devalloc_alpha_scalar_, {1});
        loaded_kernels_.update_alpha_kernel->launch(&host_ctx_);
        // add(v, v, alpha, p0)
        host_ctx_.set_arg_devalloc(0, devalloc_v_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg_devalloc(1, devalloc_v_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg<float>(2, 1.0f);
        host_ctx_.set_arg_devalloc(3, devalloc_alpha_scalar_, {1});
        host_ctx_.set_arg_devalloc(4, devalloc_p0_, {N_VERTS}, {3, 1});
        loaded_kernels_.add_scalar_ndarray_kernel->launch(&host_ctx_);
        // add(r0, r0, -alpha, mul_ans)
        host_ctx_.set_arg_devalloc(0, devalloc_r0_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg_devalloc(1, devalloc_r0_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg<float>(2, -1.0f);
        host_ctx_.set_arg_devalloc(3, devalloc_alpha_scalar_, {1});
        host_ctx_.set_arg_devalloc(4, devalloc_mul_ans_, {N_VERTS}, {3, 1});
        loaded_kernels_.add_scalar_ndarray_kernel->launch(&host_ctx_);

        // r_2_new = dot(r0, r0)
        host_ctx_.set_arg_devalloc(0, devalloc_r0_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg_devalloc(1, devalloc_r0_, {N_VERTS}, {3, 1});
        loaded_kernels_.dot2scalar_kernel->launch(&host_ctx_);

        host_ctx_.set_arg_devalloc(0, devalloc_beta_scalar_, {1});
        loaded_kernels_.update_beta_r_2_kernel->launch(&host_ctx_);

        // add(p0, r0, beta, p0)
        host_ctx_.set_arg_devalloc(0, devalloc_p0_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg_devalloc(1, devalloc_r0_, {N_VERTS}, {3, 1});
        host_ctx_.set_arg<float>(2, 1.0f);
        host_ctx_.set_arg_devalloc(3, devalloc_beta_scalar_, {1});
        host_ctx_.set_arg_devalloc(4, devalloc_p0_, {N_VERTS}, {3, 1});
        loaded_kernels_.add_scalar_ndarray_kernel->launch(&host_ctx_);
      }

      // fill_ndarray(f, 0)
      host_ctx_.set_arg_devalloc(0, devalloc_f_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg<float>(1, 0);
      loaded_kernels_.fill_ndarray_kernel->launch(&host_ctx_);

      // add(x, x, dt, v)
      host_ctx_.set_arg_devalloc(0, devalloc_x_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg_devalloc(1, devalloc_x_, {N_VERTS}, {3, 1});
      host_ctx_.set_arg<float>(2, DT);
      host_ctx_.set_arg_devalloc(3, devalloc_v_, {N_VERTS}, {3, 1});
      loaded_kernels_.add_kernel->launch(&host_ctx_);
    }
    // floor_bound(x, v)
    host_ctx_.set_arg_devalloc(0, devalloc_x_, {N_VERTS}, {3, 1});
    host_ctx_.set_arg_devalloc(1, devalloc_v_, {N_VERTS}, {3, 1});
    loaded_kernels_.floor_bound_kernel->launch(&host_ctx_);
    vulkan_runtime_->synchronize();

    // Render elements
    auto stream = device_->get_graphics_stream();
    auto cmd_list = stream->new_command_list();
    bool color_clear = true;
    std::vector<float> clear_colors = {0.03, 0.05, 0.08, 1};
    auto semaphore = surface_->acquire_next_image();
    auto image = surface_->get_target_image();
    cmd_list->begin_renderpass(
        /*xmin=*/0, /*ymin=*/0, /*xmax=*/width_,
        /*ymax=*/height_, /*num_color_attachments=*/1, &image, &color_clear,
        &clear_colors, &depth_allocation_,
        /*depth_clear=*/true);

    RenderConstants* constants =
        (RenderConstants*)device_->map(render_constants_);
    constants->proj = glm::perspective(
        glm::radians(55.0f), float(width_) / float(height_), 0.1f, 10.0f);
    constants->proj[1][1] *= -1.0f;
#ifdef ANDROID
    constexpr float kCameraZ = 4.85f;
#else
    constexpr float kCameraZ = 4.8f;
#endif
    constants->view = glm::lookAt(glm::vec3(0.0, 0.0, kCameraZ),
                                  glm::vec3(0, 0, 0), glm::vec3(0, 1.0, 0));
    device_->unmap(render_constants_);

    // Draw box
    {
      auto resource_binder = render_box_pipeline_->resource_binder();
      resource_binder->buffer(0, 0, render_constants_.get_ptr(0));
      resource_binder->vertex_buffer(devalloc_box_verts_.get_ptr(0));
      resource_binder->index_buffer(devalloc_box_indices_.get_ptr(0), 32);

      cmd_list->bind_pipeline(render_box_pipeline_.get());
      cmd_list->bind_resources(resource_binder);
      cmd_list->draw_indexed(cornell_box_indicies_.size());
    }
    // Draw mesh
    {
      auto resource_binder = render_mesh_pipeline_->resource_binder();
      resource_binder->buffer(0, 0, render_constants_.get_ptr(0));
      resource_binder->vertex_buffer(devalloc_x_.get_ptr(0));
      resource_binder->index_buffer(devalloc_indices_.get_ptr(0), 32);

      cmd_list->bind_pipeline(render_mesh_pipeline_.get());
      cmd_list->bind_resources(resource_binder);
      cmd_list->draw_indexed(N_FACES * 3);
    }

    cmd_list->end_renderpass();
    stream->submit_synced(cmd_list.get(), {semaphore});

    surface_->present_image();
  }

  void cleanup() {
    device_->dealloc_memory(devalloc_x_);
    device_->dealloc_memory(devalloc_v_);
    device_->dealloc_memory(devalloc_f_);
    device_->dealloc_memory(devalloc_mul_ans_);
    device_->dealloc_memory(devalloc_c2e_);
    device_->dealloc_memory(devalloc_b_);
    device_->dealloc_memory(devalloc_r0_);
    device_->dealloc_memory(devalloc_p0_);
    device_->dealloc_memory(devalloc_indices_);
    device_->dealloc_memory(devalloc_vertices_);
    device_->dealloc_memory(devalloc_edges_);
    device_->dealloc_memory(devalloc_ox_);
    device_->dealloc_memory(devalloc_alpha_scalar_);
    device_->dealloc_memory(devalloc_beta_scalar_);

    device_->dealloc_memory(devalloc_box_indices_);
    device_->dealloc_memory(devalloc_box_verts_);
    device_->dealloc_memory(render_constants_);
    device_->destroy_image(depth_allocation_);
  }

 private:
  struct RenderConstants {
    glm::mat4 proj;
    glm::mat4 view;
  };

  struct ImplicitFemKernels {
    taichi::lang::aot::Kernel* init_kernel{nullptr};
    taichi::lang::aot::Kernel* get_vertices_kernel{nullptr};
    taichi::lang::aot::Kernel* get_indices_kernel{nullptr};
    taichi::lang::aot::Kernel* get_force_kernel{nullptr};
    taichi::lang::aot::Kernel* advect_kernel{nullptr};
    taichi::lang::aot::Kernel* floor_bound_kernel{nullptr};
    taichi::lang::aot::Kernel* get_b_kernel{nullptr};
    taichi::lang::aot::Kernel* matmul_cell_kernel{nullptr};
    taichi::lang::aot::Kernel* ndarray_to_ndarray_kernel{nullptr};
    taichi::lang::aot::Kernel* fill_ndarray_kernel{nullptr};
    taichi::lang::aot::Kernel* add_ndarray_kernel{nullptr};
    taichi::lang::aot::Kernel* dot_kernel{nullptr};
    taichi::lang::aot::Kernel* add_kernel{nullptr};
    taichi::lang::aot::Kernel* update_alpha_kernel{nullptr};
    taichi::lang::aot::Kernel* update_beta_r_2_kernel{nullptr};
    taichi::lang::aot::Kernel* add_scalar_ndarray_kernel{nullptr};
    taichi::lang::aot::Kernel* dot2scalar_kernel{nullptr};
    taichi::lang::aot::Kernel* init_r_2_kernel{nullptr};
    taichi::lang::aot::Kernel* get_matrix_kernel{nullptr};
    taichi::lang::aot::Kernel* clear_field_kernel{nullptr};
    taichi::lang::aot::Kernel* matmul_edge_kernel{nullptr};
  };

  std::vector<uint64_t> host_result_buffer_;
  std::unique_ptr<taichi::lang::vulkan::VulkanDeviceCreator> embedded_device_{
      nullptr};
  taichi::lang::vulkan::VulkanDevice* device_{nullptr};
  std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime_{nullptr};
  std::unique_ptr<taichi::lang::aot::Module> module_{nullptr};
  ImplicitFemKernels loaded_kernels_;
  taichi::lang::RuntimeContext host_ctx_;

  std::vector<ColorVertex> cornell_box_vertices_;
  std::vector<int> cornell_box_indicies_;

  int width_{0};
  int height_{0};

  taichi::lang::DeviceAllocation devalloc_x_;
  taichi::lang::DeviceAllocation devalloc_v_;
  taichi::lang::DeviceAllocation devalloc_f_;
  taichi::lang::DeviceAllocation devalloc_mul_ans_;
  taichi::lang::DeviceAllocation devalloc_c2e_;
  taichi::lang::DeviceAllocation devalloc_b_;
  taichi::lang::DeviceAllocation devalloc_r0_;
  taichi::lang::DeviceAllocation devalloc_p0_;
  taichi::lang::DeviceAllocation devalloc_indices_;
  taichi::lang::DeviceAllocation devalloc_vertices_;
  taichi::lang::DeviceAllocation devalloc_edges_;
  taichi::lang::DeviceAllocation devalloc_ox_;
  taichi::lang::DeviceAllocation devalloc_alpha_scalar_;
  taichi::lang::DeviceAllocation devalloc_beta_scalar_;

  std::unique_ptr<taichi::lang::Surface> surface_{nullptr};
  std::unique_ptr<taichi::lang::Pipeline> render_box_pipeline_{nullptr};
  std::unique_ptr<taichi::lang::Pipeline> render_mesh_pipeline_{nullptr};
  taichi::lang::DeviceAllocation devalloc_box_verts_;
  taichi::lang::DeviceAllocation devalloc_box_indices_;
  taichi::lang::DeviceAllocation depth_allocation_;
  taichi::lang::DeviceAllocation render_constants_;
};
#endif

struct Module_implicit_fem {
  TiAotModule module_;
  TiKernel kernel_init_ = nullptr;
  TiKernel kernel_get_vertices_ = nullptr;
  TiKernel kernel_get_indices_ = nullptr;
  TiKernel kernel_get_force_ = nullptr;
  TiKernel kernel_advect_ = nullptr;
  TiKernel kernel_floor_bound_ = nullptr;
  TiKernel kernel_get_b_ = nullptr;
  TiKernel kernel_matmul_cell_ = nullptr;
  TiKernel kernel_ndarray_to_ndarray_ = nullptr;
  TiKernel kernel_fill_ndarray_ = nullptr;
  TiKernel kernel_add_ndarray_ = nullptr;
  TiKernel kernel_dot_ = nullptr;
  TiKernel kernel_add_ = nullptr;
  TiKernel kernel_update_alpha_ = nullptr;
  TiKernel kernel_update_beta_r_2_ = nullptr;
  TiKernel kernel_add_scalar_ndarray_ = nullptr;
  TiKernel kernel_dot2scalar_ = nullptr;
  TiKernel kernel_init_r_2_ = nullptr;
  TiKernel kernel_get_matrix_ = nullptr;
  TiKernel kernel_clear_field_ = nullptr;
  TiKernel kernel_matmul_edge_ = nullptr;

  TiNdShape shape_1_ { {1}, 1 };
  TiNdShape shape_N_VERTS_ { {N_VERTS}, 1 };
  TiNdShape shape_N_EDGES_ { {N_EDGES}, 1 };
  TiNdShape shape_N_CELLS_ { {N_CELLS}, 1 };
  TiNdShape elem_shape_2_1_ { {2, 1}, 2 };
  TiNdShape elem_shape_3_1_ { {3, 1}, 2 };
  TiNdShape elem_shape_4_1_ { {4, 1}, 2 };
  TiNdShape elem_shape_6_1_ { {6, 1}, 2 };

  Module_implicit_fem(const char* path) :
    module_(tiLoadVulkanAotModule(path)),
    kernel_init_(tiGetAotModuleKernel(&module_, "init")),
    kernel_floor_bound_(tiGetAotModuleKernel(&module_, "floor_bound")),
    kernel_get_b_(tiGetAotModuleKernel(&module_, "get_b")),
    kernel_matmul_cell_(tiGetAotModuleKernel(&module_, "matmul_cell")),
    kernel_ndarray_to_ndarray_(tiGetAotModuleKernel(&module_, "ndarray_to_ndarray")),
    kernel_fill_ndarray_(tiGetAotModuleKernel(&module_, "fill_ndarray")),
    kernel_add_ndarray_(tiGetAotModuleKernel(&module_, "add_ndarray")),
    kernel_add_(tiGetAotModuleKernel(&module_, "add")),
    kernel_update_alpha_(tiGetAotModuleKernel(&module_, "update_alpha")),
    kernel_update_beta_r_2_(tiGetAotModuleKernel(&module_, "update_beta_r_2")),
    kernel_add_scalar_ndarray_(tiGetAotModuleKernel(&module_, "add_scalar_ndarray")),
    kernel_dot2scalar_(tiGetAotModuleKernel(&module_, "dot2scalar")),
    kernel_init_r_2_(tiGetAotModuleKernel(&module_, "init_r_2")),
    kernel_get_matrix_(tiGetAotModuleKernel(&module_, "get_matrix")),
    kernel_clear_field_(tiGetAotModuleKernel(&module_, "clear_field")),
    kernel_matmul_edge_(tiGetAotModuleKernel(&module_, "matmul_edge"))
  {
  }

  void init(
    TiContext context,
    const TiNdArray& x,
    const TiNdArray& v,
    const TiNdArray& f,
    const TiNdArray& ox,
    const TiNdArray& vertices
  ) const {
    tiSetContextArgumentNdArray(context, 0, &x);
    tiSetContextArgumentNdArray(context, 1, &v);
    tiSetContextArgumentNdArray(context, 2, &f);
    tiSetContextArgumentNdArray(context, 3, &ox);
    tiSetContextArgumentNdArray(context, 4, &vertices);
    tiLaunchKernel(context, kernel_init_);
  }
  void floor_bound(
    TiContext context,
    const TiNdArray& x,
    const TiNdArray& v
  ) {
    tiSetContextArgumentNdArray(context, 0, &x);
    tiSetContextArgumentNdArray(context, 1, &v);
    tiLaunchKernel(context, kernel_floor_bound_);
  }
  void get_force(
    TiContext context,
    const TiNdArray& x,
    const TiNdArray& f,
    const TiNdArray& vertices,
    float g_x,
    float g_y,
    float g_z
  ) {
    tiSetContextArgumentNdArray(context, 0, &x);
    tiSetContextArgumentNdArray(context, 1, &f);
    tiSetContextArgumentNdArray(context, 2, &vertices);
    tiSetContextArgumentF32(context, 3, g_x);
    tiSetContextArgumentF32(context, 4, g_y);
    tiSetContextArgumentF32(context, 5, g_z);
    tiLaunchKernel(context, kernel_get_force_);
  }
  void get_b(
    TiContext context,
    const TiNdArray& v,
    const TiNdArray& b,
    const TiNdArray& f
  ) {
    tiSetContextArgumentNdArray(context, 0, &v);
    tiSetContextArgumentNdArray(context, 1, &b);
    tiSetContextArgumentNdArray(context, 2, &f);
    tiLaunchKernel(context, kernel_get_b_);
  }
  void ndarray_to_ndarray(
    TiContext context,
    const TiNdArray& p0,
    const TiNdArray& r0
  ) {
    tiSetContextArgumentNdArray(context, 0, &p0);
    tiSetContextArgumentNdArray(context, 1, &r0);
    tiLaunchKernel(context, kernel_ndarray_to_ndarray_);
  }
  void fill_ndarray(
    TiContext context,
    const TiNdArray& ndarray,
    float val
  ) {
    tiSetContextArgumentNdArray(context, 0, &ndarray);
    tiSetContextArgumentF32(context, 1, val);
    tiLaunchKernel(context, kernel_fill_ndarray_);
  }
  void add(
    TiContext context,
    const TiNdArray& ans,
    const TiNdArray& a,
    float k,
    const TiNdArray& b
  ) {
    tiSetContextArgumentNdArray(context, 0, &ans);
    tiSetContextArgumentNdArray(context, 1, &a);
    tiSetContextArgumentF32(context, 2, k);
    tiSetContextArgumentNdArray(context, 3, &b);
    tiLaunchKernel(context, kernel_add_);
  }
  void update_alpha(
    TiContext context,
    const TiNdArray& alpha_scalar
  ) {
    tiSetContextArgumentNdArray(context, 0, &alpha_scalar);
    tiLaunchKernel(context, kernel_update_alpha_);
  }
  void update_beta_r_2(
    TiContext context,
    const TiNdArray& beta_scalar
  ) {
    tiSetContextArgumentNdArray(context, 0, &beta_scalar);
    tiLaunchKernel(context, kernel_update_beta_r_2_);
  }
  void add_scalar_ndarray(
    TiContext context,
    const TiNdArray& ans,
    const TiNdArray& a,
    float k,
    const TiNdArray& scalar,
    const TiNdArray& b
  ) {
    tiSetContextArgumentNdArray(context, 0, &ans);
    tiSetContextArgumentNdArray(context, 1, &a);
    tiSetContextArgumentF32(context, 2, k);
    tiSetContextArgumentNdArray(context, 3, &scalar);
    tiSetContextArgumentNdArray(context, 4, &b);
    tiLaunchKernel(context, kernel_add_scalar_ndarray_);
  }
  void dot2scalar(
    TiContext context,
    const TiNdArray& r0
  ) {
    tiSetContextArgumentNdArray(context, 0, &r0);
    tiSetContextArgumentNdArray(context, 1, &r0);
    tiLaunchKernel(context, kernel_dot2scalar_);
  }
  void init_r_2(
    TiContext context
  ) {
    tiLaunchKernel(context, kernel_init_r_2_);
  }
  void get_matrix(
    TiContext context,
    const TiNdArray& c2e,
    const TiNdArray& vertices
  ) {
    tiSetContextArgumentNdArray(context, 0, &c2e);
    tiSetContextArgumentNdArray(context, 1, &vertices);
    tiLaunchKernel(context, kernel_get_matrix_);
  }
  void clear_field(TiContext context) {
    tiLaunchKernel(context, kernel_clear_field_);
  }
  void matmul_edge(
    TiContext context,
    const TiNdArray& ret,
    const TiNdArray& vel,
    const TiNdArray& edges
  ) {
    tiSetContextArgumentNdArray(context, 0, &ret);
    tiSetContextArgumentNdArray(context, 1, &vel);
    tiSetContextArgumentNdArray(context, 2, &edges);
    tiLaunchKernel(context, kernel_matmul_edge_);
  }
};

struct FemApp {
  std::unique_ptr<Module_implicit_fem> implicit_fem_;

  void run_init(int width, int height, const char* path_prefix) {
    implicit_fem_ = std::make_unique<Module_implicit_fem>(path_prefix);
    
  }
  void run_render_loop(float g_x = 0, float g_y = -9.8, float g_z = 0) {

  }
  void cleanup() {

  }
};
