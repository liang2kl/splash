#pragma once

#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace splash::ops {

// One layer's SwiGLU projections, affine Q4 or quantized GGUF tensors:
// down(silu(gate x) * up x).
struct SwiGluProjections final {
  const Projection *gate = nullptr;
  const Projection *up = nullptr;
  const Projection *down = nullptr;
};

// The dense FFN of a prefill command's chunks split by intermediate channel
// between the GPU and the Neural Engine. The GPU runs the leading channels with
// the Q4 prefill kernels; the ANE runs the rest as one W8A8 program over a
// chunk of kChunkRows rows, with int8 weights the GPU requantizes from the
// Q4 planes one layer ahead into double-buffered surfaces. The GPU adds the
// ANE's partial down projection to its own when it joins the chunk. Each
// chunk's shared event orders its evaluation between the GPU's input packing
// and that join, within the one command, and the GPU runs the command's other
// chunks in between: QwenTarget joins a chunk's FFN only before that chunk's
// next layer.
class AneFfn final {
public:
  // The rows of a prefill chunk (SPLASH_PREFILL_TOKEN_BUDGET), and the fewest
  // a command needs for the split to beat the GPU alone; smaller commands
  // keep the whole FFN on the GPU.
  static constexpr uint32_t kChunkRows = 2048;
  static constexpr uint32_t kMinimumRows = 512;

  // `share` is the fraction of intermediate channels the ANE takes, for
  // commands of up to `chunks` chunks.
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         double share, uint32_t chunks = 1);
  ~AneFfn();
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  // The Metal memory of the split of `layers` at `share` and `chunks`.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers, double share,
                                             uint32_t chunks = 1);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }

  // Starts encoding a command, discarding the jobs of one never submitted.
  void begin() noexcept;
  // Starts layer `layer`'s FFN of chunk `chunk` of the command's `chunks`, of
  // `rows` rows: output = residual + FFN of `normalized` (whose Q4 sums the
  // norm wrote), complete once addJoin(chunk). QwenTarget encodes the chunks
  // of each layer in order, from layer 0, and joins a chunk before starting
  // its next layer. The scratch buffers are the prefill arena's dense FFN
  // buffers.
  void add(metal::CommandGraph &graph, uint32_t layer, uint32_t chunk, uint32_t chunks, metal::MetalBuffer normalized,
           metal::MetalBuffer sums, metal::MetalBuffer gateScratch, metal::MetalBuffer intermediate,
           metal::MetalBuffer downSums, metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows,
           LinearScratch scratch);
  // Waits for chunk `chunk`'s ANE evaluation and adds its partial down
  // projection to the output; nothing when it has none pending.
  void addJoin(metal::CommandGraph &graph, uint32_t chunk);

  // Queues the ANE evaluations of the command encoded since the last submit,
  // before the command is committed, and returns the command's number for
  // finish(): the first at once, each later one as an earlier one completes.
  // Once one cannot be queued, it and those after it never run; their events
  // are raised when those queued have completed, so that the commands waiting
  // on them complete.
  [[nodiscard]] uint64_t submit();
  // After command `command` has completed: waits for its evaluations to
  // report, and throws if one failed or never ran.
  void finish(uint64_t command);

private:
  struct Weights final {
    // The ANE's rows of gate and up in two input segments, down in segments
    // of its inputs, and the shared per-row scale of each.
    std::vector<ane::Surface> gate, up, down;
    ane::Surface gateScale, upScale, downScale;
  };
  struct Layer final {
    SwiGluProjections source;
    // The GPU's share: gate and up views of the leading rows, and down's
    // leading inputs repacked.
    Projection gate, up, down;
  };
  // A chunk's surfaces of the ANE program, bound for each weight set, the
  // event its evaluations wait on and signal, and its FFN started and not yet
  // joined: the event value its evaluation signals, and its output rows.
  struct Slot final {
    std::vector<ane::Surface> inputs;
    ane::Surface tokenScale, partial;
    std::array<std::vector<ane::Surface>, 2> bindings;
    metal::SharedEvent event;
    uint64_t value = 0, pending = 0;
    metal::MetalBuffer output;
    uint32_t rows = 0;
  };
  struct Job final {
    uint32_t slot, set;
    uint64_t wait, signal;
  };
  // The submitted evaluations, numbered from 1 in submission order: at most
  // kQueueWindow queued to the ANE, which refuses more than 127, and the rest
  // waiting. They complete in order. One feeder at a time queues them.
  struct Queue final {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> waiting, running;
    uint64_t submitted = 0, completed = 0;
    // The numbers of the completed evaluations that failed or never ran.
    std::vector<uint64_t> failures;
    bool feeding = false, abandoned = false;
  };

  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, uint32_t part) const;
  // Queues waiting evaluations while fewer than kQueueWindow run, with the
  // queue's mutex held by `lock`.
  void feed(std::unique_lock<std::mutex> &lock);
  // Waits until evaluation `last` has completed, or none has completed for
  // kCompletionTimeout.
  [[nodiscard]] bool wait(std::unique_lock<std::mutex> &lock, uint64_t last);

  metal::MetalBackend &backend_;
  const Linear &linear_;
  uint32_t hidden_ = 0, intermediate_ = 0, gpuChannels_ = 0, aneChannels_ = 0;
  std::vector<uint32_t> downSegments_;
  std::vector<Layer> layers_;
  metal::MetalBuffer signs_, rowScales_, rotated_;
  std::array<Weights, 2> sets_;
  std::unique_ptr<ane::Program> program_;
  std::vector<Slot> slots_;
  // Encoded but not submitted.
  std::vector<Job> jobs_;
  std::shared_ptr<Queue> queue_ = std::make_shared<Queue>();
  uint64_t allocatedBytes_ = 0;
};

} // namespace splash::ops
