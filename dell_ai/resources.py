"""Port and GPU resource management for local deployments.

Supports both NVIDIA (nvidia-smi / --gpus) and AMD (rocm-smi / --device renderD)
backends. All public functions detect the available backend automatically and
degrade gracefully when neither tool is present.
"""

import re
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from typing import List, Literal, Optional

_PORT_SCAN_START = 8080
_PORT_SCAN_END = 9000

# Standard renderD base offset on ROCm systems: GPU 0 → renderD128, GPU 1 → renderD129, …
_AMD_RENDER_D_BASE = 128

GpuBackend = Optional[Literal["nvidia", "amd"]]


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_gpu_backend() -> GpuBackend:
    """Return 'nvidia', 'amd', or None depending on which SMI tool is present."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi"):
        return "amd"
    return None


# ---------------------------------------------------------------------------
# NVIDIA backend
# ---------------------------------------------------------------------------


def _nvidia_total() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return 0
        return sum(1 for line in proc.stdout.strip().splitlines() if line.strip())
    except Exception:
        return 0


def _nvidia_free_indices() -> List[int]:
    try:
        all_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if all_proc.returncode != 0:
            return []
        all_indices = [
            int(line.strip())
            for line in all_proc.stdout.strip().splitlines()
            if line.strip().isdigit()
        ]

        apps_proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        busy: set = set()
        if apps_proc.returncode == 0:
            for line in apps_proc.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    busy.add(int(line))

        return [i for i in all_indices if i not in busy]
    except Exception:
        return []


def _nvidia_parse_gpu_count(snippet: str, total_gpus: Optional[int]) -> Optional[int]:
    m = re.search(r"--gpus\s+'([^']+)'", snippet)
    if not m:
        m = re.search(r'--gpus\s+"([^"]+)"', snippet)
    if not m:
        m = re.search(r"--gpus\s+(\S+)", snippet)
    if not m:
        return None
    val = m.group(1).strip("'\"")
    if val == "all":
        return total_gpus if total_gpus is not None else _nvidia_total()
    device_m = re.search(r"device=([0-9,]+)", val)
    if device_m:
        return len(device_m.group(1).split(","))
    try:
        return int(val)
    except ValueError:
        return None


def _nvidia_inject_devices(snippet: str, indices: List[int]) -> str:
    device_str = ",".join(str(i) for i in indices)
    return re.sub(
        r"--gpus\s+(?:'[^']*'|\"[^\"]*\"|\S+)",
        f"""--gpus '"device={device_str}"'""",
        snippet,
    )


# ---------------------------------------------------------------------------
# AMD backend
# ---------------------------------------------------------------------------


def _amd_total() -> int:
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return 0
        # Lines like "GPU[0]  : GPU ID: 0x73bf"
        return sum(
            1 for line in proc.stdout.splitlines() if re.match(r"\s*GPU\[", line)
        )
    except Exception:
        return 0


def _amd_free_indices() -> List[int]:
    total = _amd_total()
    if total == 0:
        return []
    all_indices = list(range(total))
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showpids"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return all_indices
        busy: set = set()
        for line in proc.stdout.splitlines():
            # Lines like "GPU[0]  : PID 12345 - Name python3"
            m = re.match(r"\s*GPU\[(\d+)\]\s*:.*PID", line)
            if m:
                busy.add(int(m.group(1)))
        return [i for i in all_indices if i not in busy]
    except Exception:
        return all_indices


def _amd_parse_gpu_count(snippet: str) -> Optional[int]:
    # Whole /dev/dri folder means all GPUs on the machine
    if re.search(r"--device=/dev/dri(?!/)", snippet):
        return _amd_total()
    # Explicit renderD paths — one flag per GPU
    devices = re.findall(r"--device=/dev/dri/renderD\d+", snippet)
    return len(devices) if devices else None


def _amd_inject_devices(snippet: str, indices: List[int]) -> str:
    render_flags = " ".join(
        f"--device=/dev/dri/renderD{_AMD_RENDER_D_BASE + i}" for i in indices
    )

    # Case 1: snippet uses --device=/dev/dri (whole folder, all GPUs)
    # Match /dev/dri NOT followed by a slash (i.e. not a subdirectory path)
    if re.search(r"--device=/dev/dri(?!/)", snippet):
        return re.sub(
            r"--device=/dev/dri(?!/)\s*",
            f"{render_flags} ",
            snippet,
            count=1,
        ).strip()

    # Case 2: explicit renderD flags — strip old ones, re-anchor on --device=/dev/kfd.
    # --device=/dev/kfd is always present alongside renderD flags in ROCm snippets.
    stripped = re.sub(r"\s*--device=/dev/dri/renderD\d+", "", snippet)
    return stripped.replace(
        "--device=/dev/kfd", f"--device=/dev/kfd {render_flags}", 1
    ).strip()


# ---------------------------------------------------------------------------
# Public GPU API (backend-agnostic)
# ---------------------------------------------------------------------------


def get_total_gpu_count() -> int:
    """Return total GPU count, or 0 when no supported SMI tool is present."""
    backend = _detect_gpu_backend()
    match backend:
        case "nvidia":
            return _nvidia_total()
        case "amd":
            return _amd_total()
    return 0


def get_free_gpu_indices() -> List[int]:
    """Return indices of GPUs with no active compute processes."""
    backend = _detect_gpu_backend()
    match backend:
        case "nvidia":
            return _nvidia_free_indices()
        case "amd":
            return _amd_free_indices()
    return []


def allocate_gpu_indices(n: int) -> List[int]:
    """
    Reserve n free GPU indices.

    Raises RuntimeError if not enough GPUs are available. Returns an empty
    list when n is 0 or no supported SMI tool is present (graceful degradation).
    """
    if n <= 0:
        return []
    if _detect_gpu_backend() is None:
        return []

    free = get_free_gpu_indices()
    if len(free) < n:
        total = get_total_gpu_count()
        raise RuntimeError(
            f"Insufficient GPUs: {n} required, {len(free)} free out of {total} total"
        )
    return free[:n]


def parse_gpu_count(snippet: str, total_gpus: Optional[int] = None) -> Optional[int]:
    """Infer GPU count from the deployment snippet.

    Used for goodput scenarios where ``num_gpus`` is not explicitly provided.
    - NVIDIA: reads ``--gpus all/N/"device=…"``
    - AMD: counts ``--device=/dev/dri/renderD*`` flags, or returns total GPU
      count when the whole ``/dev/dri`` folder is used
    """
    backend = _detect_gpu_backend()
    match backend:
        case "nvidia":
            return _nvidia_parse_gpu_count(snippet, total_gpus)
        case "amd":
            return _amd_parse_gpu_count(snippet)
    return None


def inject_gpu_devices(snippet: str, indices: List[int]) -> str:
    """Replace generic GPU flags with an explicit device assignment.

    - NVIDIA: replaces ``--gpus all/N/"device=…"`` with ``--gpus "device=0,2"``
    - AMD: replaces ``--device=/dev/dri`` or ``--device=/dev/dri/renderD*`` flags
      with explicit renderD paths pinned to the given GPU indices
    """
    backend = _detect_gpu_backend()
    match backend:
        case "nvidia":
            return _nvidia_inject_devices(snippet, indices)
        case "amd":
            return _amd_inject_devices(snippet, indices)
    return snippet


# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------


def _is_port_free(port: int) -> bool:
    """Return True if the port is not bound by any local process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_free_port(preferred: Optional[int] = None) -> int:
    """Return a free TCP port, preferring ``preferred`` if available."""
    if preferred is not None and _is_port_free(preferred):
        return preferred
    for port in range(_PORT_SCAN_START, _PORT_SCAN_END):
        if _is_port_free(port):
            return port
    raise RuntimeError(
        f"No free port found in range {_PORT_SCAN_START}–{_PORT_SCAN_END}"
    )


def parse_host_port(snippet: str) -> Optional[int]:
    """Parse the host-side port from a Docker ``-p HOST:CONTAINER`` flag."""
    m = re.search(r"-p\s+(\d+):", snippet)
    return int(m.group(1)) if m else None


def inject_host_port(snippet: str, port: int) -> str:
    """Replace the host port in a Docker ``-p HOST:CONTAINER`` flag."""
    return re.sub(r"(-p\s+)\d+(:\d+)", rf"\g<1>{port}\2", snippet)


# ---------------------------------------------------------------------------
# Container image tag injection
# ---------------------------------------------------------------------------

# Matches the DEH image reference and stops at whitespace or quotes, so it works
# for bare Docker image args and quoted/unquoted Kubernetes manifest values alike.
_DEH_IMAGE_RE = re.compile(r"(?:registry\.dell\.huggingface\.co|cr\.hf\.co)/[^\s\"']+")
_DOCKER_DEH_IMAGE_RE = re.compile(
    r"(?<!\S)(?P<quote>[\"']?)(?:registry\.dell\.huggingface\.co|cr\.hf\.co)/"
    r"[^\s\"']+(?P=quote)"
)


def inject_image_tag(snippet: str, tag: str) -> str:
    """Set the tag on every DEH container image reference in the snippet.

    The Dell Enterprise Hub image (from the legacy registry or ``cr.hf.co``) is
    returned untagged by the API. This appends ``:tag``, replacing an existing
    tag if one is present. Works for both Docker commands and Kubernetes
    manifests, and updates every occurrence so all references stay consistent.
    """

    def _set_tag(match: "re.Match[str]") -> str:
        image = match.group(0)
        # The registry host carries no port, so any ':' present is an existing tag.
        base = image.split("@", 1)[0].split(":", 1)[0]
        return f"{base}:{tag}"

    return _DEH_IMAGE_RE.sub(_set_tag, snippet)


# ---------------------------------------------------------------------------
# Local weights / HF cache injection
# ---------------------------------------------------------------------------

CONTAINER_MODEL_PATH = "/data"
CONTAINER_HF_CACHE_PATH = "/root/.cache/huggingface"


def _inject_volume_mount(snippet: str, host_path: str, container_path: str) -> str:
    """Insert ``-v host_path:container_path`` before the DEH image reference."""
    volume = shlex.quote(f"{host_path}:{container_path}")
    return _DOCKER_DEH_IMAGE_RE.sub(
        lambda match: f"-v {volume} {match.group(0)}",
        snippet,
        count=1,
    )


def _replace_model_id(snippet: str, new_value: str) -> str:
    """Replace the MODEL_ID value in ``-e`` / ``--env`` flags. No-op if absent."""
    return re.sub(
        r"((?:-e|--env)[ \t]+)MODEL_ID=\S+",
        r"\1" + f"MODEL_ID={new_value}",
        snippet,
    )


def inject_local_dir(snippet: str, local_dir: str) -> str:
    """Add a ``-v`` mount for local model weights and redirect ``MODEL_ID``.

    Mounts *local_dir* as ``/data`` inside the container and rewrites
    ``MODEL_ID=/data`` so the inference server loads weights from disk instead
    of downloading them at runtime. No-op for non-Docker snippets.
    """
    if "docker run" not in snippet:
        return snippet
    abs_dir = str(Path(local_dir).resolve())
    snippet = _inject_volume_mount(snippet, abs_dir, CONTAINER_MODEL_PATH)
    snippet = _replace_model_id(snippet, CONTAINER_MODEL_PATH)
    return snippet


def inject_hf_cache_dir(snippet: str, hf_cache_dir: str) -> str:
    """Add a ``-v`` mount for a HuggingFace cache directory and set ``HF_HUB_CACHE``.

    Mounts *hf_cache_dir* as ``/root/.cache/huggingface`` and injects
    ``-e HF_HUB_CACHE=/root/.cache/huggingface`` so the container runtime
    finds the locally cached model files without downloading them. ``MODEL_ID``
    is left unchanged. No-op for non-Docker snippets.
    """
    if "docker run" not in snippet:
        return snippet
    abs_dir = str(Path(hf_cache_dir).resolve())
    snippet = _inject_volume_mount(snippet, abs_dir, CONTAINER_HF_CACHE_PATH)
    return _DOCKER_DEH_IMAGE_RE.sub(
        lambda match: f"-e HF_HUB_CACHE={CONTAINER_HF_CACHE_PATH} {match.group(0)}",
        snippet,
        count=1,
    )
