"""Utils for many things aroung"""


def get_batch_size(n_rows: int) -> int:
    if n_rows >= 5_120_000:
        return 4096
    if n_rows >= 1_280_000:
        return 2048
    if n_rows >= 160_000:
        return 1024
    return 256 if n_rows <= 20_000 else 512
