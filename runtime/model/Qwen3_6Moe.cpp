#include "Qwen3_6Moe.hpp"
#include "model/GgufTarget.hpp"
#include <cstring>

#include <string_view>

namespace splash::model {
namespace {

constexpr std::string_view kHeadMagic = "MDFM0002";

void requireLayout(const Qwen3_6MoeLayout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.attentionQueryHeads || !layout.attentionKvHeads ||
      !layout.attentionHeadDimension || !layout.rotaryPairs ||
      !(layout.rotaryTheta > 0.0F) || !layout.fullAttentionPeriod ||
      !layout.experts || !layout.expertsPerToken ||
      !layout.expertIntermediateSize) {
    throw WeightStoreError("Qwen3.6 MoE layout contains a zero dimension");
  }
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.attentionWidth !=
          layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.packedFullWidth !=
          2 * layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.expertsPerToken > layout.experts ||
      layout.hiddenCaptureLayers.back() >= layout.layers ||
      !layout.q8Layout().valid() || !layout.gdnStateLayout().valid()) {
    throw WeightStoreError("Qwen3.6 MoE layout is inconsistent");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

} // namespace

Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      bool ggufTarget,
                      Qwen3_6MoeLayout layout) {
  requireLayout(layout);
  const auto readFfn = [&](WeightFile &file, Qwen3_6MoeLayerWeights &layer) {
    if (ggufTarget) {
      // Section order of gguf::ImagePlanner::layer for MoE layers.
      ops::MoeGgufWeights g;
      g.router = file.section(uint64_t{layout.experts} * layout.hiddenSize * 2, "router-bf16");
      g.sharedGate = file.section(uint64_t{layout.hiddenSize} * 4, "shared-expert-scalar-gate-f32");
      g.expertGate = readGgufExpertProjection(file, layout.experts, "experts-gate");
      g.expertUp = readGgufExpertProjection(file, layout.experts, "experts-up");
      g.expertDown = readGgufExpertProjection(file, layout.experts, "experts-down");
      g.sharedExpertGate = readGgufExpertProjection(file, 1, "shared-expert-gate");
      g.sharedExpertUp = readGgufExpertProjection(file, 1, "shared-expert-up");
      g.sharedExpertDown = readGgufExpertProjection(file, 1, "shared-expert-down");
      layer.ffn.gguf = std::move(g);
      return;
    }
    layer.ffn.router = readQ8Projection(
        file, backend, layout.experts, layout.hiddenSize, "router");
    layer.ffn.expertGate = readExpertQ4Projection(
        file, layout.experts, layout.expertIntermediateSize,
        layout.hiddenSize, "experts-gate");
    layer.ffn.expertUp = readExpertQ4Projection(
        file, layout.experts, layout.expertIntermediateSize,
        layout.hiddenSize, "experts-up");
    layer.ffn.expertDown = readExpertQ4Projection(
        file, layout.experts, layout.hiddenSize,
        layout.expertIntermediateSize, "experts-down");
    layer.ffn.sharedGate = readExpertQ4Projection(
        file, 1, layout.expertIntermediateSize, layout.hiddenSize,
        "shared-expert-gate");
    layer.ffn.sharedUp = readExpertQ4Projection(
        file, 1, layout.expertIntermediateSize, layout.hiddenSize,
        "shared-expert-up");
    layer.ffn.sharedDown = readExpertQ4Projection(
        file, 1, layout.hiddenSize, layout.expertIntermediateSize,
        "shared-expert-down");
    layer.ffn.sharedExpertGate = readQ8Projection(
        file, backend, kQ4StorageN, layout.hiddenSize,
        "shared-expert-scalar-gate");
  };
  if (!ggufTarget)
    return loadQwenTargetWeights<Qwen3_6MoeWeights>(backend, directory, layout, kHeadMagic, readFfn);

  // The target directory holds the llama.cpp GGUF; every layer image is repacked into memory as it is read.
  gguf::TargetGeometry geometry;
  geometry.architecture = "qwen35moe";
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.intermediateSize = 0;
  geometry.gdnKeyHeads = layout.gdnKeyHeads;
  geometry.gdnValueHeads = layout.gdnValueHeads;
  geometry.gdnHeadDimension = layout.gdnHeadDimension;
  geometry.convolutionDimension = layout.convolutionDimension;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.fullAttentionPeriod = layout.fullAttentionPeriod;
  geometry.experts = layout.experts;
  geometry.expertsPerToken = layout.expertsPerToken;
  geometry.expertIntermediateSize = layout.expertIntermediateSize;
  GgufTargetLoader loader(backend, findTargetGguf(directory), geometry);
  struct GgufFiles {
    GgufTargetLoader &loader;
    WeightFile layer(uint32_t index, bool) { return loader.layer(index); }
    WeightFile head(uint32_t) { return loader.head(); }
    WeightFile embedding(uint32_t, uint32_t) { return loader.embedding(); }
  };
  Qwen3_6MoeWeights weights = readQwenTargetWeights<Qwen3_6MoeWeights>(
      backend, layout, GgufFiles{loader}, readFfn, true);
  // Shared scratch for the dense GGUF projections: split-K partials (8 splits x 32 rows x widest
  // projection) and their arrival counters, permuted GDN out_proj activations for the prefill
  // budget, and the grouped -> tiled value-head permutation. Expert projections run without split-K.
  const uint32_t widest = std::max({layout.packedGdnWidth, layout.packedFullWidth, layout.hiddenSize});
  metal::MetalBuffer partials = backend.allocateBuffer(
      uint64_t{8} * 32 * widest * 4, metal::BufferStorage::Shared, "gguf-partials");
  metal::MetalBuffer counters = backend.allocateBuffer(
      uint64_t{widest / 64} * 4, metal::BufferStorage::Shared, "gguf-counters");
  std::memset(counters.contents(), 0, counters.sizeBytes());
  metal::MetalBuffer permuted = backend.allocateBuffer(
      uint64_t{SPLASH_PREFILL_TOKEN_BUDGET} * layout.attentionWidth * 2,
      metal::BufferStorage::Shared, "gguf-permuted");
  const uint32_t heads = layout.gdnValueHeads, keyHeads = layout.gdnKeyHeads;
  const uint32_t perHead = heads / keyHeads;
  metal::MetalBuffer permutation = backend.allocateBuffer(
      uint64_t{heads} * 4, metal::BufferStorage::Shared, "gguf-permutation");
  auto *table = static_cast<uint32_t *>(permutation.contents());
  for (uint32_t tiled = 0; tiled < heads; ++tiled)
    table[tiled] = (tiled % keyHeads) * perHead + tiled / keyHeads;
  const auto attach = [&](ops::Q4Projection &p) {
    p.kqPartials = partials;
    p.kqCounters = counters;
    p.kqPermuted = permuted;
    p.kqPermutation = permutation;
  };
  for (Qwen3_6MoeLayerWeights &layer : weights.layers)
    std::visit([&](auto &mixer) { attach(mixer.inputProjection); attach(mixer.outputProjection); },
               layer.mixer);
  attach(weights.logitsProjection);
  return weights;
}

} // namespace splash::model
