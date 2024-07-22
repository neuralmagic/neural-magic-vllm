#pragma once

#include <torch/custom_class.h>

namespace vllm {

//
//  ScalarType can represent a wide range of floating point and integer types,
//  in particular it can be used to represent sub-byte data types (something
//  that torch.dtype currently does not support).
//
//  ScalarTypeTorch is a subclass of ScalarType that is compatible with
//  TORCH_LIBRARY, making it accessible from Python as well meaning this class
//  can be used as a argument for custom operators, helping to simplify these
//  interfaces.
//
//  The type definitions on the Python side can be found in: vllm/_core_ext.pyi
//  these type definitions should be kept up to date with any Python API changes
//  here.
//
class ScalarType {
 public:
  enum NanRepr : int64_t {
    NAN_NONE = 0,                // nans are not supported
    NAN_IEEE_754 = 1,            // nans are: exp all 1s, mantissa not all 0s
    NAN_EXTD_RANGE_MAX_MIN = 2,  // nans are: exp all 1s, mantissa all 1s

    NAN_REPR_ID_MAX
  };

  constexpr ScalarType(bool _signed, int64_t exponent, int64_t mantissa,
                       int64_t bias, bool finite_values_only = false,
                       NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        bias(bias),
        _signed(_signed),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr){};

  static constexpr ScalarType s(int64_t size_bits, int64_t bias = 0) {
    return ScalarType(true, 0, size_bits - 1, bias);
  }

  static constexpr ScalarType u(int64_t size_bits, int64_t bias = 0) {
    return ScalarType(false, 0, size_bits, bias);
  }

  // IEEE 754 compliant floating point type
  static constexpr ScalarType f(int64_t exponent, int64_t mantissa) {
    TORCH_CHECK(mantissa > 0 && exponent > 0);
    return ScalarType(true, exponent, mantissa, 0, false, NAN_IEEE_754);
  }

  // IEEE 754 non-compliant floating point type
  static constexpr ScalarType fn(int64_t exponent, int64_t mantissa,
                                 bool finite_values_only, NanRepr nan_repr) {
    TORCH_CHECK(nan_repr < NAN_REPR_ID_MAX, "Invalid NanRepr");
    TORCH_CHECK(mantissa > 0 && exponent > 0);
    TORCH_CHECK(nan_repr != NAN_IEEE_754,
                "use `f` constructor for floating point types that follow IEEE "
                "754 conventions");
    return ScalarType(true, exponent, mantissa, 0, finite_values_only,
                      nan_repr);
  }

  int64_t const exponent;  // size of the exponent field (0 for integer types)
  int64_t const mantissa;  // size of the mantissa field (size of the integer
                           // excluding the sign bit for integer types)
  int64_t const bias;      // stored values equal value + bias,
                           // used for quantized type
  bool const _signed;  // flag if the type supports negative numbers (i.e. has a
                       // sign bit)

  // Extra Floating point info
  bool const finite_values_only;  // i.e. no +/-inf if true
  NanRepr const nan_repr;         // how NaNs are represented
                                  // (not applicable for integer types)

  int64_t size_bits() const { return mantissa + exponent + is_signed(); }
  bool is_signed() const { return _signed; }
  bool is_integer() const { return exponent == 0; }
  bool is_floating_point() const { return exponent > 0; }
  bool is_ieee_754() const {
    return is_floating_point() && finite_values_only == false &&
           nan_repr == NAN_IEEE_754;
  }
  bool has_nans() const { return is_floating_point() && nan_repr != NAN_NONE; }
  bool has_infs() const {
    return is_floating_point() && finite_values_only == false;
  }
  bool has_bias() const { return bias != 0; }

 private:
  double _floating_point_max() const {
    TORCH_CHECK(mantissa <= 52 && exponent <= 11,
                "Cannot represent max/min as a double for type ", str());

    uint64_t max_mantissa = (uint64_t(1) << mantissa) - 1;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN) {
      max_mantissa -= 1;
    }

    uint64_t max_exponent = (uint64_t(1) << exponent) - 2;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN || nan_repr == NAN_NONE) {
      TORCH_CHECK(exponent < 11,
                  "Cannot represent max/min as a double for type ", str());
      max_exponent += 1;
    }

    // adjust the exponent to match that of a double
    //  for now we assume the exponent bias is the standard 2^(e-1) -1, (where e
    //  is the exponent bits), there is some precedent for non-standard biases,
    //  example `float8_e4m3b11fnuz` here: https://github.com/jax-ml/ml_dtypes
    //  but to avoid premature over complication we are just assuming the
    //  standard exponent bias until there is a need to support non-standard
    //  biases
    uint64_t exponent_bias = (uint64_t(1) << (exponent - 1)) - 1;
    uint64_t exponent_bias_double = (uint64_t(1) << 10) - 1;  // double e = 11

    uint64_t max_exponent_double =
        max_exponent - exponent_bias + exponent_bias_double;

    // shift the mantissa into the position for a double and
    // the exponent
    uint64_t double_raw =
        (max_mantissa << (52 - mantissa)) | (max_exponent_double << 52);

    return *reinterpret_cast<double*>(&double_raw);
  }

  std::variant<int64_t, double> _raw_max() const {
    if (is_floating_point()) {
      return {_floating_point_max()};
    } else {
      TORCH_CHECK(size_bits() < 64 || size_bits() == 64 && is_signed(),
                  "Cannot represent max as a int64_t");
      return {(int64_t(1) << mantissa) - 1};
    }
  }

  std::variant<int64_t, double> _raw_min() const {
    if (is_floating_point()) {
      TORCH_CHECK(is_signed(),
                  "We currently assume all floating point types are signed");
      constexpr uint64_t sign_bit_double = (uint64_t(1) << 63);

      double max = _floating_point_max();
      uint64_t max_raw = *reinterpret_cast<uint64_t*>(&max);
      uint64_t min_raw = max_raw | sign_bit_double;
      return {*reinterpret_cast<double*>(&min_raw)};
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

 public:
  // Max representable value for this scalar type.
  // (accounting for bias if there is one)
  std::variant<int64_t, double> max() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_max());
  }

  // Min representable value for this scalar type.
  // (accounting for bias if there is one)
  std::variant<int64_t, double> min() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_min());
  }

  std::string str() const {
    /*
     * generally follows: https://github.com/jax-ml/ml_dtypes
     * for floating point types (leading f):
     *  - E_: exponent size
     *  - M_: mantissa size
     *  - no-trailing letters: means it follows IEEE 754 conventions
     *  - trailing f: means finite values only (no infinities)
     *  - trailing n: means nans are supported (non-standard encoding)
     * for integer types (leading s/u):
     *  - leading s: means signed
     *  - leading u: means unsigned
     *  - number following s/u: number of bits
     *  - bX: indicates a non-zero bias of X
     */
    if (is_floating_point()) {
      auto ret =
          "fE" + std::to_string(exponent) + "M" + std::to_string(mantissa);
      if (!is_ieee_754()) {
        if (finite_values_only) {
          ret += "f";
        }
        if (nan_repr != NAN_NONE) {
          ret += "n";
        }
      }
      return ret;
    } else {
      auto ret = ((is_signed()) ? "s" : "u") + std::to_string(size_bits());
      if (has_bias()) {
        ret += "z" + std::to_string(bias);
      }
      return ret;
    }
  }

  bool operator==(ScalarType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           bias == other.bias && _signed == other._signed &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }
};

// Create a TORCH_LIBRARY compatible version of ScalarType (i.e. inherit from
//  torch::CustomClassHolder), we use multiple inheritance here since we cannot
//  have ScalarType inherit from torch::CustomClassHolder and have a constexpr
//  constructor at the same time (torch::CustomClassHolder does not have a
//  constexpr destructor)
class ScalarTypeTorch : public torch::CustomClassHolder, public ScalarType {
 public:
  ScalarTypeTorch(int64_t exponent, int64_t mantissa, int64_t bias,
                  bool _signed)
      : ScalarType(exponent, mantissa, bias, _signed){};

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

  static SelfPtr f(int64_t exponent, int64_t mantissa) {
    return c10::make_intrusive<Self>(ScalarType::f(exponent, mantissa));
  }

  static SelfPtr fn(int64_t exponent, int64_t mantissa, bool finite_values_only,
                    int64_t nan_repr) {
    return c10::make_intrusive<Self>(ScalarType::fn(
        exponent, mantissa, finite_values_only, NanRepr(nan_repr)));
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
    bind_function(cls, "is_ieee_754", &Base::is_ieee_754);
    bind_function(cls, "has_nans", &Base::has_nans);
    bind_function(cls, "has_infs", &Base::has_infs);
    bind_function(cls, "has_bias", &Base::has_bias);

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
    bind_function(cls, "__repr__", [](SelfPtr const& self) {
      return "ScalarType." + self.get()->str();
    });

    // Bind static functions (convenience constructors)
    bind_static_function(cls, "s", &ScalarTypeTorch::s);
    bind_static_function(cls, "u", &ScalarTypeTorch::u);
    bind_static_function(cls, "f", &ScalarTypeTorch::f);
    bind_static_function(cls, "fn", &ScalarTypeTorch::fn);
  }
};

using ScalarTypeTorchPtr = c10::intrusive_ptr<ScalarTypeTorch>;

/*
 * generally follows: https://github.com/jax-ml/ml_dtypes
 * for floating point types (leading f):
 *  - E_: exponent size
 *  - M_: mantissa size
 *  - no-trailing letters: means it follows IEEE 754 conventions
 *  - trailing f: means finite values only (no infinities)
 *  - trailing n: means nans are supported (non-standard encoding)
 * for integer types (leading s/u):
 *  - leading s: means signed
 *  - leading u: means unsigned
 *  - number following s/u: number of bits
 *  - bX: indicates a non-zero bias of X
 */
static inline constexpr auto kS4 = ScalarType::s(4);
static inline constexpr auto kU4 = ScalarType::u(4);
static inline constexpr auto kS8 = ScalarType::s(8);  // int8
static inline constexpr auto kU8 = ScalarType::u(8);  // uint8
static inline constexpr auto kFE3M2fn =
    ScalarType::fn(3, 2, true, ScalarType::NAN_NONE);  // FP6
static inline constexpr auto kFE3M4fn = ScalarType::fn(
    3, 4, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);          // FP8_E3M4fn
static inline constexpr auto kFE5M2 = ScalarType::f(5, 2);    // FP8_E5M2
static inline constexpr auto kFE8M7 = ScalarType::f(8, 7);    // BFloat16
static inline constexpr auto kFE5M10 = ScalarType::f(5, 10);  // Float16

// "gptq" types
static inline constexpr auto kU4B8 = ScalarType::u(4, 8);
static inline constexpr auto kU8B128 = ScalarType::u(8, 128);

};  // namespace vllm
