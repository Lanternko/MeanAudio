# Verbatim from matchering/utils.py 2.0.6
def ms_to_samples(value: float, sample_rate: int) -> int:
    return int(sample_rate * value * 1e-3)


def make_odd(value: int) -> int:
    return value + 1 if not value & 1 else value
