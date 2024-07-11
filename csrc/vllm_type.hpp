#pragma once

#include <torch/custom_class.h>

class VLLMType {
 public:
  constexpr VLLMType(int64_t mantissa, int64_t exponent, bool _signed)
      : mantissa(mantissa), exponent(exponent), _signed(_signed){};

  int64_t const mantissa = 0;
  int64_t const exponent = 0;
  bool const _signed = true;

  int64_t size_bits() const { return mantissa + exponent + is_signed(); }
  bool is_signed() const { return _signed; }
  bool is_integer() const { return exponent == 0; }
  bool is_floating_point() const { return exponent > 0; }

  std::variant<int64_t, double> max() const {
    if (is_floating_point()) {
      // TODO: return max floating point value as double
      //   see `dequant_8bit<bfloat16>` in `csrc/quantization/fp8/fp8_marlin.cu`
      //   to see how this could be done
      TORCH_CHECK_NOT_IMPLEMENTED(is_floating_point(), "Not implemented");
      return {nan("")};
    } else {
      TORCH_CHECK(size_bits() < 64 || size_bits() == 64 && is_signed(),
                  "Cannot represent max as a int64_t");
      return {(int64_t(1) << mantissa) - 1};
    }
  }

  std::variant<int64_t, double> min() const {
    if (is_floating_point()) {
      // TODO: return min floating point value as double
      //   see `dequant_8bit<bfloat16>` in `csrc/quantization/fp8/fp8_marlin.cu`
      //   to see how this could be done
      TORCH_CHECK_NOT_IMPLEMENTED(is_floating_point(), "Not implemented");
      return {nan("")};
    } else {
      TORCH_CHECK(!is_signed() || size_bits() <= 64,
                  "Cannot represent min as a int64_t");
      if (is_signed()) {
        // set the top bit to 1 (i.e. INT64_MIN) and the rest to 0
        // then perform an arithmetic shift right to set all the bits above
        // (size_bits() - 1) to 1
        return {INT64_MIN >> (64 - size_bits())};
      } else {
        return {int64_t(0)};
      }
    }
  }

  std::string str() const {
    if (is_floating_point()) {
      auto ret =
          "fE " + std::to_string(exponent) + "M" + std::to_string(mantissa);
      if (!is_signed()) {
        ret += "u";
      }
      return ret;
    } else {
      return ((is_signed()) ? "s" : "u") + std::to_string(size_bits());
    }
  }

  bool operator==(VLLMType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           _signed == other._signed;
  }
};

class VLLMTypeTorch : public torch::CustomClassHolder, public VLLMType {
 public:
  VLLMTypeTorch(int64_t mantissa, int64_t exponent, bool _signed)
      : VLLMType(mantissa, exponent, _signed){};

  VLLMTypeTorch(VLLMType type) : VLLMType(type){};
};

using VLLMTypeTorchPtr = c10::intrusive_ptr<VLLMTypeTorch>;

// Common types
static inline constexpr VLLMType kI4(3, 0, true);
static inline constexpr VLLMType kU4(4, 0, false);
