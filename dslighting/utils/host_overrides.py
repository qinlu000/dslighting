"""Process-local DNS overrides for API endpoints with split-horizon DNS."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import threading
from typing import Any

logger = logging.getLogger(__name__)

ENV_API_HOST_OVERRIDES = "API_HOST_OVERRIDES"

_lock = threading.RLock()
_original_getaddrinfo = socket.getaddrinfo
_host_overrides: dict[str, str] = {}
_installed = False


def activate_api_host_overrides(raw: str | None = None) -> dict[str, str]:
    """Install explicit hostname-to-IP mappings for the current Python process.

    ``API_HOST_OVERRIDES`` must be a JSON object whose values are literal IPv4
    or IPv6 addresses.  Mapped hosts are also appended to ``NO_PROXY`` so HTTP
    clients resolve them locally instead of delegating DNS to a configured
    proxy.  The override is process-local and inherited only when child Python
    processes activate it from the same environment.
    """
    value = os.getenv(ENV_API_HOST_OVERRIDES) if raw is None else raw
    if value is None or not value.strip():
        return {}

    mappings = _parse_host_overrides(value)
    global _installed
    with _lock:
        _host_overrides.update(mappings)
        if not _installed:
            socket.getaddrinfo = _getaddrinfo_with_overrides
            _installed = True

    _append_no_proxy(mappings)
    logger.info(
        "Activated process-local API host override(s): %s",
        ", ".join(sorted(mappings)),
    )
    return dict(mappings)


def _parse_host_overrides(raw: str) -> dict[str, str]:
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{ENV_API_HOST_OVERRIDES} must be valid JSON") from exc

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{ENV_API_HOST_OVERRIDES} must be a non-empty JSON object")

    mappings: dict[str, str] = {}
    for host, address in payload.items():
        normalized_host = _normalize_host(host)
        if not isinstance(address, str):
            raise ValueError(
                f"{ENV_API_HOST_OVERRIDES}[{normalized_host!r}] must be an IP address"
            )
        try:
            normalized_address = str(ipaddress.ip_address(address.strip()))
        except ValueError as exc:
            raise ValueError(
                f"{ENV_API_HOST_OVERRIDES}[{normalized_host!r}] must be a literal IP address"
            ) from exc
        mappings[normalized_host] = normalized_address
    return mappings


def _normalize_host(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{ENV_API_HOST_OVERRIDES} hostnames must be strings")
    host = value.strip().rstrip(".").lower()
    if not host or any(character.isspace() for character in host) or "/" in host:
        raise ValueError(f"Invalid hostname in {ENV_API_HOST_OVERRIDES}: {value!r}")
    return host


def _getaddrinfo_with_overrides(
    host: str | bytes | None,
    port: str | int | None,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
):
    lookup_host = host
    if isinstance(host, bytes):
        try:
            normalized_host = host.decode("ascii").rstrip(".").lower()
        except UnicodeDecodeError:
            normalized_host = ""
    elif isinstance(host, str):
        normalized_host = host.rstrip(".").lower()
    else:
        normalized_host = ""

    with _lock:
        override = _host_overrides.get(normalized_host)
    if override is not None:
        lookup_host = override
    return _original_getaddrinfo(lookup_host, port, family, type, proto, flags)


def _append_no_proxy(mappings: dict[str, str]) -> None:
    for variable in ("NO_PROXY", "no_proxy"):
        entries = [
            entry.strip()
            for entry in os.environ.get(variable, "").split(",")
            if entry.strip()
        ]
        normalized_entries = {entry.rstrip(".").lower() for entry in entries}
        for host in mappings:
            if host not in normalized_entries:
                entries.append(host)
                normalized_entries.add(host)
        os.environ[variable] = ",".join(entries)


__all__ = ["ENV_API_HOST_OVERRIDES", "activate_api_host_overrides"]
