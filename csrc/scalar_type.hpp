#pragma once

#include <torch/custom_class.h>

namespace vllm {

class ScalarType {
 public:
  constexpr ScalarType(int64_t mantissa, int64_t exponent, int64_t bias,
                       bool _signed)
      : mantissa(mantissa), exponent(exponent), bias(bias), _signed(_signed){};

  static constexpr ScalarType s(int64_t size_bits, int64_t bias = 0) {
    return ScalarType(size_bits - 1, 0, bias, true);
  }

  static constexpr ScalarType u(int64_t size_bits, int64_t bias = 0) {
    return ScalarType(size_bits, 0, bias, false);
  }

  static constexpr ScalarType f(int64_t mantissa, int64_t exponent) {
    return ScalarType(mantissa, exponent, 0, true);
  }

  int64_t const mantissa = 0;
  int64_t const exponent = 0;
  int64_t const bias = 0;
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

  bool operator==(ScalarType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           _signed == other._signed;
  }
};

class ScalarTypeTorch : public torch::CustomClassHolder, public ScalarType {
 public:
  ScalarTypeTorch(int64_t mantissa, int64_t exponent, int64_t bias,
                  bool _signed)
      : ScalarType(mantissa, exponent, bias, _signed){};

  ScalarTypeTorch(ScalarType type) : ScalarType(type){};

  using Base = ScalarType;
  using Self = ScalarTypeTorch;
  using SelfPtr = c10::intrusive_ptr<Self>;

  static SelfPtr s(int64_t size_bits, c10::optional<int64_t> bias) {
    return c10::make_intrusive<Self>(
        ScalarType::s(size_bits, bias.value_or(0)));
  }

  static SelfPtr u(int64_t size_bits, c10::optional<int64_t> bias) {
    return c10::make_intrusive<Self>(
        ScalarType::u(size_bits, bias.value_or(0)));
  }

  static SelfPtr f(int64_t mantissa, int64_t exponent) {
    return c10::make_intrusive<Self>(ScalarType::f(mantissa, exponent));
  }

  template <typename T>
  static void bind_readonly_property(torch::class_<Self>& cls,
                                     std::string const& name, T Base::*field) {
    auto getter_func = [field = std::move(field)](SelfPtr const& self) {
      if constexpr (std::is_member_function_pointer_v<decltype(field)>) {
        return (self.get()->*field)();
      } else {
        return self.get()->*field;
      }
    };

    cls.def_property(name, getter_func);
  }

  template <typename MemberFunc, typename Cls>
  static void bind_function(torch::class_<Self>& cls, const std::string& name,
                            MemberFunc Cls::*member) {
    cls.def(name, [member = std::move(member)](SelfPtr const& self) {
      return (self.get()->*member)();
    });
  }

  template <typename Func>
  static void bind_function(torch::class_<Self>& cls, const std::string& name,
                            Func func) {
    cls.def(name, func);
  }

  template <typename Func>
  static void bind_static_function(torch::class_<Self>& cls,
                                   const std::string& name, Func func) {
    cls.def_static(name, func);
  }

  static void bind_class(torch::Library& lib) {
    auto cls = lib.class_<ScalarTypeTorch>("ScalarType")
                   .def(torch::init<int64_t, int64_t, int64_t, bool>());

    // Bind Properties
    bind_readonly_property(cls, "mantissa", &Base::mantissa);
    bind_readonly_property(cls, "exponent", &Base::exponent);
    bind_readonly_property(cls, "bias", &Base::bias);
    bind_readonly_property(cls, "size_bits", &Base::size_bits);

    // Bind member functions
    bind_function(cls, "is_signed", &Base::is_signed);
    bind_function(cls, "is_integer", &Base::is_integer);
    bind_function(cls, "is_floating_point", &Base::is_floating_point);
    bind_function(cls, "max", [](SelfPtr const& self) {
      return std::visit([](auto arg) { return c10::IValue(arg); },
                        self.get()->max());
    });
    bind_function(cls, "min", [](SelfPtr const& self) {
      return std::visit([](auto arg) { return c10::IValue(arg); },
                        self.get()->min());
    });
    bind_function(cls, "__str__", &Base::str);
    bind_function(cls, "__eq__", [](SelfPtr const& self, SelfPtr const& other) {
      return *self == *other;
    });
    bind_function(cls, "__repr__",
                  [](SelfPtr const& self, SelfPtr const& other) {
                    return "ScalarType." + self.get()->str();
                  });

    // Bind static functions (convience constructors)
    bind_static_function(cls, "s", &ScalarTypeTorch::s);
    bind_static_function(cls, "u", &ScalarTypeTorch::u);
    bind_static_function(cls, "f", &ScalarTypeTorch::f);
  }
};

using ScalarTypeTorchPtr = c10::intrusive_ptr<ScalarTypeTorch>;

// Common types
static inline constexpr auto kS4 = ScalarType::s(4);
static inline constexpr auto kU4 = ScalarType::u(4);
static inline constexpr auto kU4Z8 = ScalarType::u(4, 8);
};  // namespace vllm