#include "tiaot/framework.hpp"
#include "taichi/cpp/taichi.hpp"

constexpr float DT = 7.5e-3;
constexpr int NUM_SUBSTEPS = 2;
constexpr int CG_ITERS = 8;
constexpr float ASPECT_RATIO = 2.0f;

struct ImplicitFem {
  ti::Runtime runtime;

  ti::AotModule aot_module;
  ti::ComputeGraph g_init;
  ti::ComputeGraph g_substep;

  ti::NdArray<float> hes_edge;
  ti::NdArray<float> hes_vert;
  ti::NdArray<float> x;
  ti::NdArray<float> v;
  ti::NdArray<float> f;
  ti::NdArray<float> mul_ans;
  ti::NdArray<int32_t> c2e;
  ti::NdArray<float> b;
  ti::NdArray<float> r0;
  ti::NdArray<float> p0;
  ti::NdArray<int32_t> indices;
  ti::NdArray<int32_t> vertices;
  ti::NdArray<int32_t> edges;
  ti::NdArray<float> ox;
  ti::NdArray<float> alpha_scalar;
  ti::NdArray<float> beta_scalar;
  ti::NdArray<float> m;
  ti::NdArray<float> B;
  ti::NdArray<float> W;
  ti::NdArray<float> dot_ans;
  ti::NdArray<float> r_2_scalar;

  void initialize(TiArch arch, const std::string& asset_dir) {
    runtime = ti::Runtime(arch);

    aot_module = runtime.load_aot_module(asset_dir + "shaders");
    g_init = aot_module.get_compute_graph("init");
    g_substep = aot_module.get_compute_graph("substep");

    auto c2eData = tiaot::load_bin<int32_t>(asset_dir + "c2e.bin");
    auto edgesData = tiaot::load_bin<int32_t>(asset_dir + "edges.bin");
    auto indicesData = tiaot::load_bin<int32_t>(asset_dir + "indices.bin");
    auto oxData = tiaot::load_bin<float>(asset_dir + "ox.bin");
    auto verticesData = tiaot::load_bin<int32_t>(asset_dir + "vertices.bin");
    
    uint32_t vertexCount = oxData.size() / 3;
    uint32_t edgeCount = edgesData.size() / 2;
    uint32_t faceCount = indicesData.size() / 3;
    uint32_t cellCount = c2eData.size() / 6;

    hes_edge = runtime.allocate_ndarray<float>({edgeCount}, {});
    hes_vert = runtime.allocate_ndarray<float>({cellCount}, {});
    x = runtime.allocate_ndarray<float>({vertexCount}, {3});
    v = runtime.allocate_ndarray<float>({vertexCount}, {3});
    f = runtime.allocate_ndarray<float>({vertexCount}, {3});
    mul_ans = runtime.allocate_ndarray<float>({vertexCount}, {3});
    c2e = runtime.allocate_ndarray<int32_t>({cellCount}, {6}, true);
    b = runtime.allocate_ndarray<float>({vertexCount}, {3});
    r0 = runtime.allocate_ndarray<float>({vertexCount}, {3});
    p0 = runtime.allocate_ndarray<float>({vertexCount}, {3});
    indices = runtime.allocate_ndarray<int32_t>({faceCount}, {3}, true);
    vertices = runtime.allocate_ndarray<int32_t>({cellCount}, {4}, true);
    edges = runtime.allocate_ndarray<int32_t>({edgeCount}, {2}, true);
    ox = runtime.allocate_ndarray<float>({vertexCount}, {3}, true);
    alpha_scalar = runtime.allocate_ndarray<float>({}, {});
    beta_scalar = runtime.allocate_ndarray<float>({}, {});
    m = runtime.allocate_ndarray<float>({vertexCount}, {});
    B = runtime.allocate_ndarray<float>({cellCount}, {3, 3});
    W = runtime.allocate_ndarray<float>({cellCount}, {});
    dot_ans = runtime.allocate_ndarray<float>({}, {});
    r_2_scalar = runtime.allocate_ndarray<float>({}, {});
  }

  void update() {
    runtime.();
  }
};

void initialize(int argc, const char** argv, tiaot::AppConfig* cfg) {
  cfg->user_data = new ImplicitFem();
  cfg->title = "Implicit FEM";
  cfg->width = 512;
  cfg->height = 512 * ASPECT_RATIO;

  ((ImplicitFem*)cfg->user_data)->initialize(TI_ARCH_VULKAN);
}

void update(tiaot::AppData* data) {
  ((ImplicitFem*)data->user_data)->update();
}

void finalize(tiaot::AppData* data) {
  delete (ImplicitFem*)data->user_data;
}
