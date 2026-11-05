"""Hardware/software snapshot of the box the kit runs on.

Everything comes from local commands and files (nvidia-smi, /proc, /sys,
dmidecode). Each source degrades independently: a missing tool leaves its
section empty with an "error" note instead of failing the snapshot.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

GPU_FIELDS = [
    "index", "name", "uuid", "pci.bus_id", "driver_version", "vbios_version",
    "memory.total", "power.limit", "power.max_limit",
    "pcie.link.gen.current", "pcie.link.gen.max",
    "pcie.link.width.current", "pcie.link.width.max",
    "clocks.max.sm", "clocks.max.mem",
]


def run_cmd(argv: list[str], timeout: float = 20) -> tuple[bool, str]:
    if not shutil.which(argv[0]):
        return False, f"{argv[0]} not found"
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"{argv[0]} timed out"
    if p.returncode != 0:
        return False, (p.stderr or p.stdout).strip()[:500]
    return True, p.stdout


def _fix_checksum(buf: bytearray, at: int, start: int, end: int) -> None:
    buf[at] = 0
    buf[at] = -sum(buf[start:end]) & 0xFF


def dmi_dump(entry: bytes, table: bytes) -> bytes | None:
    """Build a `dmidecode --from-dump` file from the raw sysfs SMBIOS tables.

    Docker masks /sys/firmware, so the container gets the host's
    /sys/firmware/dmi/tables mounted elsewhere instead. A dump file is the
    entry point at offset 0 and the table at 32, with the entry point's
    table address pointing at 32, as `dmidecode --dump-bin` writes it.
    """
    ep = bytearray(entry)
    if ep[:5] == b"_SM3_" and len(ep) >= 24:
        ep[0x10:0x18] = (32).to_bytes(8, "little")
        _fix_checksum(ep, 0x05, 0, ep[0x06])
    elif ep[:4] == b"_SM_" and len(ep) >= 31:
        ep[0x18:0x1C] = (32).to_bytes(4, "little")
        _fix_checksum(ep, 0x15, 0x10, 0x1F)
        _fix_checksum(ep, 0x04, 0, ep[0x05])
    else:
        return None
    return bytes(ep[:32]).ljust(32, b"\0") + table


def dmidecode_from_tables(src: str) -> tuple[bool, str]:
    try:
        dump = dmi_dump(Path(src, "smbios_entry_point").read_bytes(), Path(src, "DMI").read_bytes())
    except OSError as e:
        return False, f"dmidecode: no /dev/mem and no tables at {src} ({e.strerror})"
    if dump is None:
        return False, "dmidecode: unknown SMBIOS entry point"
    with tempfile.NamedTemporaryFile(suffix=".dmi") as f:
        f.write(dump)
        f.flush()
        return run_cmd(["dmidecode", "--from-dump", f.name, "-t", "17"])


def _num(v: str):
    v = v.strip()
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return None if v in ("[N/A]", "N/A", "[Not Supported]", "") else v


def parse_gpu_csv(text: str, fields: list[str] = GPU_FIELDS) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        g = {}
        for f, v in zip(fields, parts):
            g[f.replace(".", "_")] = v if f in ("name", "uuid", "pci.bus_id", "driver_version",
                                                "vbios_version") else _num(v)
        gpus.append(g)
    return gpus


def parse_dmidecode_memory(text: str) -> list[dict]:
    """Parse `dmidecode -t 17` into populated DIMM slots."""
    dimms = []
    for block in text.split("\n\n"):
        if "Memory Device" not in block:
            continue
        d = {}
        for line in block.splitlines():
            m = re.match(r"^\s+([A-Za-z ]+):\s*(.*)$", line)
            if m:
                d[m.group(1).strip()] = m.group(2).strip()
        size = d.get("Size", "")
        if not size or size.startswith("No Module"):
            continue
        dimms.append({
            "locator": d.get("Locator"),
            "size": size,
            "type": d.get("Type"),
            "speed": d.get("Speed"),
            "configured_speed": d.get("Configured Memory Speed") or d.get("Configured Clock Speed"),
            "manufacturer": d.get("Manufacturer"),
            "part_number": d.get("Part Number"),
        })
    return dimms


def parse_cpuinfo(text: str) -> dict:
    models, phys, cores = set(), set(), set()
    threads = 0
    cur_phys = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        if k == "processor":
            threads += 1
        elif k == "model name":
            models.add(v)
        elif k == "physical id":
            cur_phys = v
            phys.add(v)
        elif k == "core id":
            cores.add((cur_phys, v))
    return {"model": " / ".join(sorted(models)) or platform.processor(),
            "sockets": len(phys) or 1, "cores": len(cores) or threads, "threads": threads}


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def snapshot(engines: list[dict] | None = None) -> dict:
    snap: dict = {"errors": {}}

    ok, out = run_cmd(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
                       "--format=csv,noheader,nounits"])
    snap["gpus"] = parse_gpu_csv(out) if ok else []
    if not ok:
        snap["errors"]["gpus"] = out
    ok, out = run_cmd(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", out) if ok else None
    snap["cuda_driver_version"] = m.group(1) if m else None
    ok, out = run_cmd(["nvidia-smi", "topo", "-m"])
    snap["gpu_topology"] = out.strip() if ok else None

    cpuinfo = _read("/proc/cpuinfo") or ""
    snap["cpu"] = parse_cpuinfo(cpuinfo)
    meminfo = _read("/proc/meminfo") or ""
    m = re.search(r"MemTotal:\s+(\d+)", meminfo)
    snap["memory"] = {"total_gb": round(int(m.group(1)) / 1024 / 1024, 1) if m else None}
    ok, out = run_cmd(["dmidecode", "-t", "17"])
    if not ok:
        ok, out = dmidecode_from_tables(os.environ.get("BENCHKIT_DMI_DIR", "/host-dmi"))
    if ok:
        dimms = parse_dmidecode_memory(out)
        snap["memory"]["dimms"] = dimms
        types = sorted({d["type"] for d in dimms if d.get("type")})
        speeds = sorted({d["configured_speed"] for d in dimms if d.get("configured_speed")})
        snap["memory"]["type"] = "/".join(types) or None
        snap["memory"]["configured_speed"] = "/".join(speeds) or None
        snap["memory"]["populated_slots"] = len(dimms)
    else:
        snap["errors"]["dimms"] = out
    snap["board"] = {k: _read(f"/sys/class/dmi/id/{k}") for k in
                     ("board_vendor", "board_name", "bios_version", "sys_vendor", "product_name")}
    snap["os"] = {"kernel": platform.release(), "machine": platform.machine(),
                  "in_container": os.path.exists("/.dockerenv")}
    snap["engines"] = engines or []
    return snap
