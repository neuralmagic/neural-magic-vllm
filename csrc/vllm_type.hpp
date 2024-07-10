#pragma once

#include <torch/custom_class.h>

class VLLMType {
 public:
  constexpr VLLMType(int64_t mantissa, int64_t exponent, bool _signed)
      : mantissa(mantissa), exponent(exponent), _signed(_signed){};

  int64_t const mantissa = 0;
  int64_t const exponent = 0;
  bool const _signed = true;

  int64_t size_bits() const { return mantissa + exponent + _signed; }
  int64_t integer() const { return exponent == 0; }
  int64_t floating_point() const { return exponent > 0; }

  std::string str() const {
    if (floating_point()) {
      auto ret =
          "fE " + std::to_string(exponent) + "M" + std::to_string(mantissa);
      if (!_signed) {
        ret += "u";
      }
      return ret;
    } else {
      return ((_signed) ? "s" : "u") + std::to_string(size_bits());
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

static inline constexpr VLLMType kInt4(3, 0, true);
static inline constexpr VLLMType kUint4(4, 0, false);
