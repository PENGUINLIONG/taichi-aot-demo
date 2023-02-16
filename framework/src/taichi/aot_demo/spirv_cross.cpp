#include <cstdint>
#include <string>
#include <vector>
#include "spirv_msl.hpp"

namespace ti {
namespace aot_demo {

std::string spv2msl(const std::vector<uint32_t> &spv) {
  spirv_cross::CompilerMSL compiler(spv.data(), spv.size());
  spirv_cross::CompilerMSL::Options options{};
  options.enable_decoration_binding = true;
  compiler.set_msl_options(options);
  std::string msl = compiler.compile();
  return msl;
}

}  // namespace ti
}  // namespace ti
