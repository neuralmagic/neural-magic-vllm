from magic_wand.data_type import DataType, DataTypeTraitsMap
import torch


def marlinv2_schedule_heuristic(
    M: int, N: int, K: int, data_type: DataType, group_size: int
):
    assert DataTypeTraitsMap[data_type].size == 4

    tile_scheduler = "void"
    if K > 4096 * 2:
        tile_scheduler = "NMstreamK"

    if M >= 128:
        tile_shape_mn = (128, 128)
    elif M > 16:
        tile_shape_mn = (128, 64)
    else:
        tile_shape_mn = (128, 16)

    cluster_shape_mnk = (1, 1, 1)

    tile_shape_mn_str = "x".join(map(str, tile_shape_mn))
    cluster_shape_mnk_str = "x".join(map(str, cluster_shape_mnk))
    schedule_name = (
        f"{tile_shape_mn_str}_{cluster_shape_mnk_str}"
        f"_TmaMI_TmaCoop_{tile_scheduler}"
    )

    return schedule_name


def marlinv2_prepack(B_quant: torch.Tensor, B_type: DataType) -> torch.Tensor:
    B_type_traits = DataTypeTraitsMap[B_type]

    return torch.ops.nm_ops.marlinv2_prepack_B(
        B_quant,
        B_type_traits.signed,
        B_type_traits.size,
    )


def marlinv2_mm(
    A: torch.Tensor,
    B_quant: torch.Tensor,
    B_type: DataType,
    B_scales: Optional[torch.Tensor] = None,
    B_zero_points: Optional[torch.Tensor] = None,
    C: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    schedule: Optional[str] = None,
    perm: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
) -> torch.Tensor:
    B_type_traits = DataTypeTraitsMap[B_type]
    size_m = A.shape[0]
    size_n = B_quant.shape[1]
    size_k = A.shape[1]

    assert perm is None and is_k_full, "Not Implemented"

    # assert size_k = B_quant.shape[0]
    assert not B_type_traits.is_floating_point

    B_group_size = None
    if B_scales is not None:
        assert size_k % B_scales.shape[0] == 0
        B_group_size = size_k // B_scales.shape[0]

    if schedule is None:
        schedule = marlinv2_schedule_heuristic(
            size_m, size_n, size_k, B_type, B_group_size
        )

    return torch.ops.nm_ops.marlinv2_mm(
        A,
        B_quant,
        B_type_traits.signed,
        B_type_traits.size,
        B_scales,
        B_zero_points,
        B_group_size,
        C,
        alpha,
        beta,
        schedule,
    )
