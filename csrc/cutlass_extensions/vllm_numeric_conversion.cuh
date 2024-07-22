// Based off of: cutlass/numeric_conversion.h

#pragma once

#include "cutlass/numeric_conversion.h"
#include "cutlass_extensions/vllm_custom_types.cuh"

namespace cutlass {

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Array<cutlass::float_e4m3_t, N> <=
/// Array<cutlass::int4b_t, N>

// TODO: Implement
#if 0 

template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<cutlass::vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  using result_type_packed_8 = Array<cutlass::float_e4m3_t, 8>;
  using result_type_packed_4 = Array<cutlass::float_e4m3_t, 4>;
  using source_type_packed_8 = Array<cutlass::vllm_uint4b8_t, 8>;
  using source_type_packed_4 = Array<cutlass::vllm_uint4b8_t, 4>;

  using ScalarConverter =
      NumericConverter<cutlass::float_e4m3_t, vllm_uint4b8_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses a lookup table to converts i4 -> e4m3.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(
      PackedSrcType const& source) {
    static_assert(
        (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
         platform::is_same<PackedResultType, result_type_packed_4>::value) ||
            (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
             platform::is_same<PackedResultType, result_type_packed_8>::value),
        "Invalid PackedSrcType/PackedResultType must be 4 or 8 to use private "
        "convert dispatch.");

    // Hold FP8 outputs in reg. We need 1 reg for every 4 outputs.
    cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4,
                          sizeof(PackedResultType)>
        r;

    // View the input as reg
    uint32_t reg = to_reg(source);

    // Determines if to get from the signed or unsigned candidates
    uint32_t sign = (reg & 0x88888888) >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = (reg & 0x77777777);

    // Signed is OR'd with 0x32103210 to find the correct value in the LUT
    const uint32_t final_prmt_base = 0x32103210;

    // [0, 1, 2, 3] encoded as FP8
    static constexpr uint32_t POS_E4M3s_REG1 = 0x44403800;
    // [4, 5, 6, 7] encoded as FP8
    static constexpr uint32_t POS_E4M3s_REG2 = 0x4E4C4A48;
    // [-1, -2, -3, -4] encoded as FP8
    static constexpr uint32_t NEG_E4M3s_REG1 = 0xCACCCED0;
    // [-5, -6, -7, -7] encoded as FP8
    static constexpr uint32_t NEG_E4M3s_REG2 = 0xB8C0C4C8;

    const int iters = PackedSrcType::kElements / 4;
  #pragma unroll
    for (int ii = 0; ii < iters; ++ii, lut_idx >>= 16, sign >>= 16) {
      uint32_t final_prmt_idx = final_prmt_base | sign;

      // This uses a look up table to convert packed int4s to packed fp8s, using
      // the int4 value as the index to prmt. It first select both the positive
      // and negative candidates, then uses the sign bit to select the correct
      // candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 pos_f8s, neg_f8s;\n"
          "  prmt.b32 pos_f8s, %1, %2, %5;\n"
          "  prmt.b32 neg_f8s, %3, %4, %5;\n"
          "  prmt.b32 %0, pos_f8s, neg_f8s, %6;\n"
          "}\n"
          : "=r"(r[ii])
          : "n"(POS_E4M3s_REG1), "n"(POS_E4M3s_REG2), "n"(NEG_E4M3s_REG1),
            "n"(NEG_E4M3s_REG2), "r"(lut_idx), "r"(final_prmt_idx));
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType =
        NumericArrayConverter<typename result_type::Element,
                              typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<
        ConverterType, result_type_packed_8, source_type_packed_8,
        result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

#endif

// for Array<cutlass::half_t, N> <= Array<vllm_uint4b8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<vllm_uint4b8_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  using result_type_packed_8 = Array<cutlass::half_t, 8>;
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using result_type_packed_2 = Array<cutlass::half_t, 2>;
  using source_type_packed_8 = Array<vllm_uint4b8_t, 8>;
  using source_type_packed_4 = Array<vllm_uint4b8_t, 4>;
  using source_type_packed_2 = Array<vllm_uint4b8_t, 2>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses bit tricks to construct a known FP16 number, then
  // does a subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(
      PackedSrcType const& source) {
    static_assert(
        (platform::is_same<PackedSrcType, source_type_packed_2>::value &&
         platform::is_same<PackedResultType, result_type_packed_2>::value) ||
            (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
             platform::is_same<PackedResultType,
                               result_type_packed_4>::value) ||
            (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
             platform::is_same<PackedResultType, result_type_packed_8>::value),
        "Invalid PackedSrcType/PackedResultType must be 2, 4 or 8 to use "
        "private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray =
        cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                              sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);

    // Below constructs the following temporary:
    // fp16s_01 = {0x00, i4_01, 0x00, i4_01}
    // fp16s_23 = {0x00, i4_23, 0x00, i4_23}
    // fp16s_45 = {0x00, i4_45, 0x00, i4_45}
    // fp16s_67 = {0x00, i4_67, 0x00, i4_67}
    // We use inline asm instead of __byte_perm intrinsic since we don't want
    // the documented (& 0x7) on the index. NVCC might be able to optimize it
    // out since the index is a constexpr, but we choose to be safe about it
    // here.
    uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
    static_assert(RegArray::kElements <= 4,
                  "Too many inputs for F16 -> I4 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  prmt.b32 %0, %1, %2, %3;\n"
          "}\n"
          : "=r"(r[ii])
          : "r"(src_reg), "n"(0), "r"(prmt_indices[ii]));
    }

    // Since the stored 4bit values are biased by 8 we get stored_val = (x+8)
    //  we are trying to construct x and a fp16 value
    // The below XOR does the following:
    //  1) Sets the exponent bits of the FP16 to the correct value for the FP16
    //  magic_num. We will be constructing {1024+16*(x1+8), 1024+(x0+8)}, where
    //  x1 in the high nibble and x0 is the low nibble then using hfma to
    //  subtract 1032 from that
    // The AND does the following:
    //  1) Clear the set bits for the int4 we will ignore.
    // We use lop3 so that we can use 1 instruction for AND and XOR.
    static constexpr uint32_t xor_mask = 0x64006400;
    static constexpr uint32_t and_mask = 0xFFF0FF0F;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask) ^ xor_mask
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // We will issue 2 hfmas that do the following:
    // {x1, x0} = {1024+16*(x1+8), 1024+(x0+8)} * {1/16, 1} - {72, 1032}
    //          = {x1 + 1152, x0 + 1032} * {1/16, 1} - {72, 1032}
    static constexpr uint32_t hfma_bias_rep = 0xD480E408;   // {72, 1032}
    static constexpr uint32_t hfma_scale_rep = 0x2C003C00;  // {1 / 16, 1}

    const half2& hfma_bias = reinterpret_cast<const half2&>(hfma_bias_rep);
    const half2& hfma_scale = reinterpret_cast<const half2&>(hfma_scale_rep);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
      fp16x2_val = __hfma2(hfma_scale, fp16x2_val, hfma_bias);
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType =
        NumericArrayConverter<typename result_type::Element,
                              typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<
        ConverterType, result_type_packed_8, source_type_packed_8,
        result_type_packed_4, source_type_packed_4, result_type_packed_2,
        source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// for Array<cutlass::bfloat16_t, N> <= Array<vllm_uint4b8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, vllm_uint4b8_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<vllm_uint4b8_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  using result_type_packed_8 = Array<cutlass::bfloat16_t, 8>;
  using result_type_packed_4 = Array<cutlass::bfloat16_t, 4>;
  using result_type_packed_2 = Array<cutlass::bfloat16_t, 2>;
  using source_type_packed_8 = Array<vllm_uint4b8_t, 8>;
  using source_type_packed_4 = Array<vllm_uint4b8_t, 4>;
  using source_type_packed_2 = Array<vllm_uint4b8_t, 2>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses bit tricks to construct a known FP16 number, then
  // does a subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(
      PackedSrcType const& source) {
    static_assert(
        (platform::is_same<PackedSrcType, source_type_packed_2>::value &&
         platform::is_same<PackedResultType, result_type_packed_2>::value) ||
            (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
             platform::is_same<PackedResultType,
                               result_type_packed_4>::value) ||
            (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
             platform::is_same<PackedResultType, result_type_packed_8>::value),
        "Invalid PackedSrcType/PackedResultType must be 2, 4 or 8 to use "
        "private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray =
        cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2,
                              sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 4;

    // Below constructs the following temporary:
    uint32_t const prmt_indices[4] = {0xF4F0, 0xF5F1, 0xF6F2, 0xF7F3};
    static_assert(RegArray::kElements <= 4,
                  "Too many inputs for BF16 -> I4 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  prmt.b32 %0, %1, %2, %3;\n"
          "}\n"
          : "=r"(r[ii])
          : "r"(src_reg), "r"(src_reg_shifted), "r"(prmt_indices[ii]));
    }

    // Since the stored 4bit values are biased by 8 we get stored_val = (x+8)
    //  we are trying to construct x and a BF16 value
    // The below XOR does the following:
    //  1) Sets the exponent bits of the BF16 to the correct value for the BF16
    //  magic_num. We will be constructing {128 + (x1+8), 128 + (x0+8)}
    //  and subtracting 136 to get {x1, x0}
    static constexpr uint32_t xor_mask = 0x43004300;
    static constexpr uint32_t and_mask = 0x000F000F;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask) ^ xor_mask
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // We will issue 2 bfmas that do the following:
    // high BF16:
    // hi_bf16 - 136, lo_bf16 - 136

    // This is the BF16 {136, 136} represented as an integer.
    static constexpr uint32_t bias_rep = 0x43084308;
    const __nv_bfloat162& bias =
        reinterpret_cast<const __nv_bfloat162&>(bias_rep);

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
      bf16x2_val = __hsub2(bf16x2_val, bias);
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType =
        NumericArrayConverter<typename result_type::Element,
                              typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<
        ConverterType, result_type_packed_8, source_type_packed_8,
        result_type_packed_4, source_type_packed_4, result_type_packed_2,
        source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

#endif

}  // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
