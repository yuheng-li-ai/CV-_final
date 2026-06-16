from cvhw3_scene.gpu import GpuMemory, parse_nvidia_smi_csv, select_freest_gpu


def test_parse_nvidia_smi_csv_and_selects_most_free_gpu():
    rows = parse_nvidia_smi_csv(
        """
        0, 1000, 49140
        1, 200, 49140
        2, 30000, 49140
        """
    )

    assert rows == [
        GpuMemory(index=0, used_mb=1000, total_mb=49140),
        GpuMemory(index=1, used_mb=200, total_mb=49140),
        GpuMemory(index=2, used_mb=30000, total_mb=49140),
    ]
    assert select_freest_gpu(rows) == "cuda:1"


def test_select_gpu_tie_breaks_by_lowest_index():
    assert (
        select_freest_gpu(
            [
                GpuMemory(index=2, used_mb=100, total_mb=1000),
                GpuMemory(index=0, used_mb=100, total_mb=1000),
            ]
        )
        == "cuda:0"
    )
