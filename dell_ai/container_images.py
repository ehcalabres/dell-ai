"""Pull Hugging Face registry images into the local Docker image store."""

import os
import shlex
import shutil
import subprocess
from typing import Optional

from dell_ai.exceptions import AuthenticationError

LEGACY_REGISTRY = "registry.dell.huggingface.co"
XET_REGISTRY = "cr.hf.co"
DEH_REGISTRIES = (LEGACY_REGISTRY, XET_REGISTRY)


def get_registry_image(command: str) -> Optional[str]:
    """Find the DEH image argument in a Docker deployment command."""
    for token in shlex.split(command):
        if token.startswith(tuple(f"{host}/" for host in DEH_REGISTRIES)):
            return token
    return None


def ensure_hf_image() -> None:
    """Install hf-image when the HF CLI cannot execute the pull command."""
    if not shutil.which("hf"):
        raise RuntimeError(
            "The Hugging Face CLI (hf) is required for cr.hf.co images. "
            "Install it with `pip install --upgrade huggingface_hub`, "
            "then run `hf extensions install hf-image`."
        )

    probe = subprocess.run(
        ["hf", "image", "pull", "--help"], capture_output=True, text=True
    )
    if probe.returncode == 0:
        return

    try:
        subprocess.run(
            ["hf", "extensions", "install", "hf-image"],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["hf", "image", "pull", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "Could not install or run hf-image. Update the Hugging Face CLI "
            "with `hf update`, then run `hf extensions install hf-image`."
            + (f"\n\n{detail}" if detail else "")
        ) from exc


def pull_xet_image(
    image: str, token: Optional[str], platform: Optional[str] = None
) -> None:
    """Authenticate and pull an image without exposing the token in argv."""
    if not token:
        raise AuthenticationError(
            "Pulling images from cr.hf.co requires a Hugging Face token. "
            "Run `dell-ai login` before deploying."
        )
    ensure_hf_image()
    command = ["hf", "image", "pull", image]
    if platform:
        command.extend(["--platform", platform])
    pull_env = os.environ.copy()
    pull_env["HF_TOKEN"] = token
    subprocess.run(command, env=pull_env, capture_output=True, text=True, check=True)
