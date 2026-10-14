def get_device() -> str:
    """torch 디바이스 우선순위: cuda > mps > cpu.
    torch는 여기서만 import한다: torch와 pycolmap은 각자 OpenMP 런타임을 들고 있어 한 프로세스에 같이 올라오면
    abort된다 — detect.postprocess를 쓰는 recon3d(pycolmap)가 torch를 끌어오지 않게."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    print(get_device())
