"""Deployment registry for tracking active model and app deployments."""

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dell_ai.container_images import LEGACY_REGISTRY, XET_REGISTRY

DEH_IMAGE_PREFIX = f"{LEGACY_REGISTRY}/enterprise-dell-inference-"


def get_global_deployments_path() -> Path:
    """Get the path to the global deployments registry file."""
    return Path.home() / ".config" / "dell-ai" / "deployments.json"


def get_local_deployments_path() -> Path:
    """Get the path to the local (current working directory) deployments file."""
    return Path.cwd() / ".dell-ai-deployments.json"


def load_deployments_file(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_deployments_file(path: Path, deployments: Dict[str, Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(deployments, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save deployments file at {path}: {str(e)}")


def save_deployment(
    deployment_id: str, metadata: Dict[str, Any], is_global: bool = False
) -> None:
    path = get_global_deployments_path() if is_global else get_local_deployments_path()
    deployments = load_deployments_file(path)

    if "deployed_at" not in metadata:
        metadata["deployed_at"] = datetime.utcnow().isoformat() + "Z"

    deployments[deployment_id] = metadata
    save_deployments_file(path, deployments)


def get_deployment(
    deployment_id: str, is_global: Optional[bool] = None
) -> Optional[Dict[str, Any]]:
    if is_global is True:
        return load_deployments_file(get_global_deployments_path()).get(deployment_id)
    elif is_global is False:
        return load_deployments_file(get_local_deployments_path()).get(deployment_id)
    else:
        local = load_deployments_file(get_local_deployments_path())
        if deployment_id in local:
            return local[deployment_id]
        return load_deployments_file(get_global_deployments_path()).get(deployment_id)


def delete_deployment(deployment_id: str, is_global: bool = False) -> bool:
    path = get_global_deployments_path() if is_global else get_local_deployments_path()
    deployments = load_deployments_file(path)
    if deployment_id in deployments:
        del deployments[deployment_id]
        save_deployments_file(path, deployments)
        return True
    return False


def _discover_docker_deployments() -> Dict[str, Dict[str, Any]]:
    """
    Scan running Docker containers for DEH images not tracked in the registry.

    Identifies legacy DEH images, Dell-tagged Xet images, and tracked Xet
    containers. Returns metadata keyed by the image name without its tag.
    """
    if not shutil.which("docker"):
        return {}

    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return {}
    except Exception:
        return {}

    tracked = {
        **load_deployments_file(get_global_deployments_path()),
        **load_deployments_file(get_local_deployments_path()),
    }
    tracked_ids = {
        meta["container_id"][:12]
        for meta in tracked.values()
        if meta.get("container_id")
    }
    discovered: Dict[str, Dict[str, Any]] = {}
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            container = json.loads(line)
        except Exception:
            continue

        image = container.get("Image", "")
        container_id = container.get("ID", "")
        is_legacy = image.startswith(DEH_IMAGE_PREFIX)
        is_xet = image.startswith(f"{XET_REGISTRY}/")
        tag = image.rsplit(":", 1)[-1] if ":" in image and "@" not in image else ""
        if not (
            is_legacy
            or (
                is_xet and (tag.startswith("dell-") or container_id[:12] in tracked_ids)
            )
        ):
            continue

        # Parse first exposed host port: "0.0.0.0:8080->80/tcp, ..." → 8080
        ports_str = container.get("Ports", "")
        endpoint = None
        port_match = re.search(r"(?:0\.0\.0\.0|::):(\d+)->", ports_str)
        if port_match:
            endpoint = f"http://localhost:{port_match.group(1)}"

        # Deployment ID based on image + unique identifier.
        prefix = DEH_IMAGE_PREFIX if is_legacy else f"{XET_REGISTRY}/"
        slug = image[len(prefix) :].split("@", 1)[0].split(":", 1)[0]
        unique_slug = slug
        counter = 1
        while unique_slug in discovered:
            unique_slug = f"{slug}_{counter}"
            counter += 1

        discovered[unique_slug] = {
            "container_id": container_id,
            "endpoint": endpoint,
            "engine": "docker",
            "image": image,
            "discovered_at": datetime.utcnow().isoformat() + "Z",
        }

    return discovered


def get_unique_deployment_id(base_id: str) -> str:
    """
    Return a deployment ID that doesn't collide with any existing registry entry.

    If base_id is already taken, appends _1, _2, … until a free slot is found.
    """
    existing = set(list_deployments().keys())
    if base_id not in existing:
        return base_id
    counter = 1
    while f"{base_id}_{counter}" in existing:
        counter += 1
    return f"{base_id}_{counter}"


def list_deployments(is_global: Optional[bool] = None) -> Dict[str, Dict[str, Any]]:
    """
    List all deployments, merging the registry with live Docker discovery.

    Running DEH containers not yet in the registry are auto-registered so they
    can be listed and managed (e.g. via undeploy) immediately.
    """
    if is_global is True:
        registry = load_deployments_file(get_global_deployments_path())
    elif is_global is False:
        registry = load_deployments_file(get_local_deployments_path())
    else:
        global_deps = load_deployments_file(get_global_deployments_path())
        local_deps = load_deployments_file(get_local_deployments_path())
        registry = {**global_deps, **local_deps}

    # Discover running DEH containers (docker ps without -a = running only).
    discovered = _discover_docker_deployments()
    running_ids = {
        meta.get("container_id", "")[:12]
        for meta in discovered.values()
        if meta.get("container_id")
    }

    # Remove registry entries for Docker containers that are no longer running.
    # Skip this when docker isn't installed so we never purge entries we can't verify.
    if shutil.which("docker"):
        stale = [
            dep_id
            for dep_id, meta in registry.items()
            if meta.get("engine") == "docker"
            and meta.get("container_id")
            and meta["container_id"][:12] not in running_ids
        ]
        for dep_id in stale:
            del registry[dep_id]
            if is_global is True:
                delete_deployment(dep_id, is_global=True)
            elif is_global is False:
                delete_deployment(dep_id, is_global=False)
            else:
                delete_deployment(dep_id, is_global=False)
                delete_deployment(dep_id, is_global=True)

    # Register any newly discovered containers not yet tracked in the registry.
    known_container_ids = {
        v.get("container_id", "")[:12]
        for v in registry.values()
        if v.get("container_id")
    }
    for slug, meta in discovered.items():
        if meta.get("container_id", "")[:12] not in known_container_ids:
            save_deployment(slug, meta, is_global=is_global is True)
            registry[slug] = meta

    return registry
