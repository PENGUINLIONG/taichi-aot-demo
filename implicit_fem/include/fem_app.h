#pragma once

#define TI_WITH_VULKAN 1
#include "taichi/taichi.h"
#include "glm/glm.hpp"
#include "volk.h"

#include <memory>
#include <vector>

#include "box_color_data.h"
#include "mesh_data.h"

constexpr float DT = 7.5e-3;
constexpr int NUM_SUBSTEPS = 2;
constexpr int CG_ITERS = 8;
constexpr float ASPECT_RATIO = 2.0f;

void load_data(TiDevice device,
  TiDeviceMemory devalloc, const void* data,
  size_t size) {
  void* mapped = tiMapMemory(device, devalloc);
  std::memcpy(mapped, data, size);
  tiUnmapMemory(device, devalloc);
}

template<typename T>
TiNdArray alloc_ndarray(TiDevice device, const std::vector<uint32_t>& shape, const std::vector<uint32_t>& elem_shape, TiMemoryUsageFlags extra_usage = 0) {
  size_t size = sizeof(T);

  TiNdShape shape2;
  for (size_t i = 0; i < shape.size(); ++i) {
    uint32_t dim = shape.at(i);
    shape2.dims[i] = dim;
    size *= dim;
  }
  shape2.dimCount = shape.size();

  TiNdShape elem_shape2;
  for (size_t i = 0; i < elem_shape.size(); ++i) {
    uint32_t dim = shape.at(i);
    elem_shape2.dims[i] = dim;
    size *= dim;
  }
  elem_shape2.dimCount = elem_shape.size();

  TiMemoryAllocateInfo mai {};
  mai.hostWrite = true;
  mai.size = size;
  mai.usage = TI_MEMORY_USAGE_STORAGE_BIT | extra_usage;

  TiNdArray out {};
  out.devmem = tiAllocateMemory(device, &mai);
  out.shape = std::move(shape2);
  out.elem_shape = std::move(elem_shape2);
  return out;
}

void free_ndarray(TiDevice device, TiNdArray& ndarray) {
  tiFreeMemory(device, ndarray.devmem);
}

struct Module_implicit_fem {
  TiContext context_ = TI_NULL_HANDLE;
  TiAotModule module_ = TI_NULL_HANDLE;
  TiKernel kernel_init_ = TI_NULL_HANDLE;
  TiKernel kernel_get_vertices_ = TI_NULL_HANDLE;
  TiKernel kernel_get_indices_ = TI_NULL_HANDLE;
  TiKernel kernel_get_force_ = TI_NULL_HANDLE;
  TiKernel kernel_advect_ = TI_NULL_HANDLE;
  TiKernel kernel_floor_bound_ = TI_NULL_HANDLE;
  TiKernel kernel_get_b_ = TI_NULL_HANDLE;
  TiKernel kernel_matmul_cell_ = TI_NULL_HANDLE;
  TiKernel kernel_ndarray_to_ndarray_ = TI_NULL_HANDLE;
  TiKernel kernel_fill_ndarray_ = TI_NULL_HANDLE;
  TiKernel kernel_add_ndarray_ = TI_NULL_HANDLE;
  TiKernel kernel_dot_ = TI_NULL_HANDLE;
  TiKernel kernel_add_ = TI_NULL_HANDLE;
  TiKernel kernel_update_alpha_ = TI_NULL_HANDLE;
  TiKernel kernel_update_beta_r_2_ = TI_NULL_HANDLE;
  TiKernel kernel_add_scalar_ndarray_ = TI_NULL_HANDLE;
  TiKernel kernel_dot2scalar_ = TI_NULL_HANDLE;
  TiKernel kernel_init_r_2_ = TI_NULL_HANDLE;
  TiKernel kernel_get_matrix_ = TI_NULL_HANDLE;
  TiKernel kernel_clear_field_ = TI_NULL_HANDLE;
  TiKernel kernel_matmul_edge_ = TI_NULL_HANDLE;

  Module_implicit_fem(TiContext context, const char* module_path) :
    module_(tiLoadVulkanAotModule(context, module_path)),
    kernel_init_(tiGetAotModuleKernel(module_, "init")),
    kernel_floor_bound_(tiGetAotModuleKernel(module_, "floor_bound")),
    kernel_get_b_(tiGetAotModuleKernel(module_, "get_b")),
    kernel_matmul_cell_(tiGetAotModuleKernel(module_, "matmul_cell")),
    kernel_ndarray_to_ndarray_(tiGetAotModuleKernel(module_, "ndarray_to_ndarray")),
    kernel_fill_ndarray_(tiGetAotModuleKernel(module_, "fill_ndarray")),
    kernel_add_ndarray_(tiGetAotModuleKernel(module_, "add_ndarray")),
    kernel_add_(tiGetAotModuleKernel(module_, "add")),
    kernel_update_alpha_(tiGetAotModuleKernel(module_, "update_alpha")),
    kernel_update_beta_r_2_(tiGetAotModuleKernel(module_, "update_beta_r_2")),
    kernel_add_scalar_ndarray_(tiGetAotModuleKernel(module_, "add_scalar_ndarray")),
    kernel_dot2scalar_(tiGetAotModuleKernel(module_, "dot2scalar")),
    kernel_init_r_2_(tiGetAotModuleKernel(module_, "init_r_2")),
    kernel_get_matrix_(tiGetAotModuleKernel(module_, "get_matrix")),
    kernel_clear_field_(tiGetAotModuleKernel(module_, "clear_field")),
    kernel_matmul_edge_(tiGetAotModuleKernel(module_, "matmul_edge")) {}
  ~Module_implicit_fem() {
    tiDestroyAotModule(module_);
  }

  void init(
    const TiNdArray& x,
    const TiNdArray& v,
    const TiNdArray& f,
    const TiNdArray& ox,
    const TiNdArray& vertices
  ) const {
    tiSetContextArgNdArray(context_, 0, &x);
    tiSetContextArgNdArray(context_, 1, &v);
    tiSetContextArgNdArray(context_, 2, &f);
    tiSetContextArgNdArray(context_, 3, &ox);
    tiSetContextArgNdArray(context_, 4, &vertices);
    tiLaunchKernel(context_, kernel_init_);
  }
  void floor_bound(
    const TiNdArray& x,
    const TiNdArray& v
  ) {
    tiSetContextArgNdArray(context_, 0, &x);
    tiSetContextArgNdArray(context_, 1, &v);
    tiLaunchKernel(context_, kernel_floor_bound_);
  }
  void get_force(
    const TiNdArray& x,
    const TiNdArray& f,
    const TiNdArray& vertices,
    float g_x,
    float g_y,
    float g_z
  ) {
    tiSetContextArgNdArray(context_, 0, &x);
    tiSetContextArgNdArray(context_, 1, &f);
    tiSetContextArgNdArray(context_, 2, &vertices);
    tiSetContextArgF32(context_, 3, g_x);
    tiSetContextArgF32(context_, 4, g_y);
    tiSetContextArgF32(context_, 5, g_z);
    tiLaunchKernel(context_, kernel_get_force_);
  }
  void get_b(
    const TiNdArray& v,
    const TiNdArray& b,
    const TiNdArray& f
  ) {
    tiSetContextArgNdArray(context_, 0, &v);
    tiSetContextArgNdArray(context_, 1, &b);
    tiSetContextArgNdArray(context_, 2, &f);
    tiLaunchKernel(context_, kernel_get_b_);
  }
  void ndarray_to_ndarray(
    const TiNdArray& p0,
    const TiNdArray& r0
  ) {
    tiSetContextArgNdArray(context_, 0, &p0);
    tiSetContextArgNdArray(context_, 1, &r0);
    tiLaunchKernel(context_, kernel_ndarray_to_ndarray_);
  }
  void fill_ndarray(
    const TiNdArray& ndarray,
    float val
  ) {
    tiSetContextArgNdArray(context_, 0, &ndarray);
    tiSetContextArgF32(context_, 1, val);
    tiLaunchKernel(context_, kernel_fill_ndarray_);
  }
  void add(
    const TiNdArray& ans,
    const TiNdArray& a,
    float k,
    const TiNdArray& b
  ) {
    tiSetContextArgNdArray(context_, 0, &ans);
    tiSetContextArgNdArray(context_, 1, &a);
    tiSetContextArgF32(context_, 2, k);
    tiSetContextArgNdArray(context_, 3, &b);
    tiLaunchKernel(context_, kernel_add_);
  }
  void update_alpha(
    const TiNdArray& alpha_scalar
  ) {
    tiSetContextArgNdArray(context_, 0, &alpha_scalar);
    tiLaunchKernel(context_, kernel_update_alpha_);
  }
  void update_beta_r_2(
    const TiNdArray& beta_scalar
  ) {
    tiSetContextArgNdArray(context_, 0, &beta_scalar);
    tiLaunchKernel(context_, kernel_update_beta_r_2_);
  }
  void add_scalar_ndarray(
    const TiNdArray& ans,
    const TiNdArray& a,
    float k,
    const TiNdArray& scalar,
    const TiNdArray& b
  ) {
    tiSetContextArgNdArray(context_, 0, &ans);
    tiSetContextArgNdArray(context_, 1, &a);
    tiSetContextArgF32(context_, 2, k);
    tiSetContextArgNdArray(context_, 3, &scalar);
    tiSetContextArgNdArray(context_, 4, &b);
    tiLaunchKernel(context_, kernel_add_scalar_ndarray_);
  }
  void dot2scalar(
    const TiNdArray& a,
    const TiNdArray& b
  ) {
    tiSetContextArgNdArray(context_, 0, &a);
    tiSetContextArgNdArray(context_, 1, &b);
    tiLaunchKernel(context_, kernel_dot2scalar_);
  }
  void init_r_2() {
    tiLaunchKernel(context_, kernel_init_r_2_);
  }
  void get_matrix(
    const TiNdArray& c2e,
    const TiNdArray& vertices
  ) {
    tiSetContextArgNdArray(context_, 0, &c2e);
    tiSetContextArgNdArray(context_, 1, &vertices);
    tiLaunchKernel(context_, kernel_get_matrix_);
  }
  void clear_field() {
    tiLaunchKernel(context_, kernel_clear_field_);
  }
  void matmul_edge(
    const TiNdArray& ret,
    const TiNdArray& vel,
    const TiNdArray& edges
  ) {
    tiSetContextArgNdArray(context_, 0, &ret);
    tiSetContextArgNdArray(context_, 1, &vel);
    tiSetContextArgNdArray(context_, 2, &edges);
    tiLaunchKernel(context_, kernel_matmul_edge_);
  }
};

struct FemApp {
  TiDevice device_;
  TiContext context_;

  TiVulkanDeviceInteropInfo interop_info_;

  std::unique_ptr<Module_implicit_fem> implicit_fem_;

  uint32_t width_;
  uint32_t height_;

  TiNdArray x_;
  TiNdArray v_;
  TiNdArray f_;
  TiNdArray mul_ans_;
  TiNdArray c2e_;
  TiNdArray b_;
  TiNdArray r0_;
  TiNdArray p0_;
  TiNdArray indices_;
  TiNdArray vertices_;
  TiNdArray edges_;
  TiNdArray ox_;
  TiNdArray alpha_scalar_;
  TiNdArray beta_scalar_;

  void run_init(int width, int height, const char* path_prefix) {
    width_ = width;
    height_ = height;

    device_ = tiCreateDevice(TI_ARCH_VULKAN);
    tiExportVulkanDevice(device_, &interop_info_);
    context_ = tiCreateContext(device_);
    implicit_fem_ = std::make_unique<Module_implicit_fem>(context_, path_prefix);

    x_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 }, TI_MEMORY_USAGE_VERTEX_BIT);
    v_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 });
    f_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 });
    mul_ans_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 });
    c2e_ = alloc_ndarray<int>(device_, { N_CELLS }, { 6, 1 });
    b_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 });
    r0_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 });
    p0_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 });
    indices_ = alloc_ndarray<int>(device_, { N_FACES }, { 3, 1 }, TI_MEMORY_USAGE_INDEX_BIT);
    vertices_ = alloc_ndarray<int>(device_, { N_CELLS }, { 4, 1 });
    edges_ = alloc_ndarray<int>(device_, { N_EDGES }, { 2, 1 });
    ox_ = alloc_ndarray<float>(device_, { N_VERTS }, { 3, 1 });
    alpha_scalar_ = alloc_ndarray<float>(device_, { 1 }, {});
    beta_scalar_ = alloc_ndarray<float>(device_, { 1 }, {});

    load_data(device_, indices_.devmem, indices_data, sizeof(indices_data));
    load_data(device_, c2e_.devmem, c2e_data, sizeof(c2e_data));
    load_data(device_, vertices_.devmem, vertices_data, sizeof(vertices_data));
    load_data(device_, ox_.devmem, ox_data, sizeof(ox_data));
    load_data(device_, edges_.devmem, edges_data, sizeof(edges_data));

    implicit_fem_->clear_field();
    implicit_fem_->init(x_, v_, f_, ox_, vertices_);
    implicit_fem_->get_matrix(c2e_, vertices_);
    tiDeviceWaitIdle(device_);
  }
  void run_render_loop(float g_x = 0, float g_y = -9.8, float g_z = 0) {
    for (int i = 0; i < NUM_SUBSTEPS; i++) {
      implicit_fem_->get_force(x_, f_, vertices_, g_x, g_y, g_z);
      implicit_fem_->get_b(v_, b_, f_);
      implicit_fem_->matmul_edge(mul_ans_, v_, edges_);
      implicit_fem_->add(r0_, b_, -1.0f, mul_ans_);
      implicit_fem_->ndarray_to_ndarray(p0_, r0_);
      implicit_fem_->dot2scalar(r0_, r0_);
      implicit_fem_->init_r_2();

      for (int i = 0; i < CG_ITERS; i++) {
        implicit_fem_->matmul_edge(mul_ans_, p0_, edges_);
        implicit_fem_->dot2scalar(p0_, mul_ans_);
        implicit_fem_->update_alpha(alpha_scalar_);
        implicit_fem_->add_scalar_ndarray(v_, v_, 1.0f, alpha_scalar_, p0_);
        implicit_fem_->dot2scalar(r0_, r0_);
        implicit_fem_->update_beta_r_2(beta_scalar_);
        implicit_fem_->add_scalar_ndarray(p0_, r0_, 1.0f, beta_scalar_, p0_);
      }

      implicit_fem_->fill_ndarray(f_, 0.0f);
      implicit_fem_->add(x_, x_, DT, v_);
    }
    implicit_fem_->floor_bound(x_, v_);
    tiDeviceWaitIdle(device_);
  }
  void cleanup() {
    free_ndarray(device_, x_);
    free_ndarray(device_, v_);
    free_ndarray(device_, f_);
    free_ndarray(device_, mul_ans_);
    free_ndarray(device_, c2e_);
    free_ndarray(device_, b_);
    free_ndarray(device_, r0_);
    free_ndarray(device_, p0_);
    free_ndarray(device_, indices_);
    free_ndarray(device_, vertices_);
    free_ndarray(device_, edges_);
    free_ndarray(device_, ox_);
    free_ndarray(device_, alpha_scalar_);
    free_ndarray(device_, beta_scalar_);
    tiDestroyContext(context_);
    tiDestroyDevice(device_);
  }
};
