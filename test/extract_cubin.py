#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def decode_mlir_string(s: str) -> bytes:
    out = bytearray()
    i = 0

    while i < len(s):
        if s[i] == "\\":
            # MLIR byte escape: \XX
            if i + 2 < len(s):
                pair = s[i + 1:i + 3]
                if re.fullmatch(r"[0-9A-Fa-f]{2}", pair):
                    out.append(int(pair, 16))
                    i += 3
                    continue

            if i + 1 < len(s) and s[i + 1] == "\\":
                out.append(ord("\\"))
                i += 2
                continue

            if i + 1 < len(s) and s[i + 1] == '"':
                out.append(ord('"'))
                i += 2
                continue

            raise RuntimeError(
                f"unsupported escape at position {i}: {s[i:i+12]!r}"
            )

        out.append(ord(s[i]))
        i += 1

    return bytes(out)


def quoted_strings(text: str, start: int):
    i = start

    while i < len(text):
        q0 = text.find('"', i)
        if q0 < 0:
            return

        j = q0 + 1
        while j < len(text):
            if text[j] == '"':
                # A quote is escaped iff preceded by an odd
                # number of backslashes.
                nslash = 0
                k = j - 1
                while k >= q0 and text[k] == "\\":
                    nslash += 1
                    k -= 1

                if nslash % 2 == 0:
                    yield text[q0 + 1:j]
                    i = j + 1
                    break
            j += 1
        else:
            raise RuntimeError("unterminated quoted string")


def elf_file_size(data: bytes, off: int) -> int:
    """
    Compute ELF file extent from ELF64 section table.
    Assumes little-endian ELF64, as CUDA cubins are.
    """
    if data[off:off + 4] != b"\x7fELF":
        raise RuntimeError("not ELF")

    elf_class = data[off + 4]
    endian = data[off + 5]

    if elf_class != 2:
        raise RuntimeError(f"expected ELF64, class={elf_class}")
    if endian != 1:
        raise RuntimeError(f"expected little-endian ELF, endian={endian}")

    def u16(pos):
        return int.from_bytes(
            data[off + pos:off + pos + 2], "little"
        )

    def u64(pos):
        return int.from_bytes(
            data[off + pos:off + pos + 8], "little"
        )

    e_shoff = u64(40)
    e_shentsize = u16(58)
    e_shnum = u16(60)

    if e_shoff == 0 or e_shnum == 0:
        raise RuntimeError("ELF has no section table")

    end = 0

    for n in range(e_shnum):
        sh = off + e_shoff + n * e_shentsize

        sh_type = int.from_bytes(
            data[sh + 4:sh + 8], "little"
        )
        sh_offset = int.from_bytes(
            data[sh + 24:sh + 32], "little"
        )
        sh_size = int.from_bytes(
            data[sh + 32:sh + 40], "little"
        )

        # SHT_NOBITS occupies no bytes in file.
        if sh_type != 8:
            end = max(end, sh_offset + sh_size)

    # Section-header table itself is also part of the ELF.
    end = max(
        end,
        e_shoff + e_shentsize * e_shnum
    )

    return end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument(
        "--binary-name",
        default="kanlib_piecewise_benchmark_kernel",
    )
    args = ap.parse_args()

    text = Path(args.input).read_text()

    marker = f"gpu.binary @{args.binary_name}"
    start = text.find(marker)

    if start < 0:
        raise RuntimeError(f"could not find {marker}")

    candidates = []

    for n, encoded in enumerate(quoted_strings(text, start)):
        try:
            data = decode_mlir_string(encoded)
        except Exception:
            continue

        elf_off = data.find(b"\x7fELF")

        # Print useful diagnostics for nontrivial strings.
        if len(data) > 16 or elf_off >= 0:
            print(
                f"quoted[{n}]: decoded={len(data)} "
                f"magic={data[:8].hex()} "
                f"elf_offset={elf_off}"
            )

        if elf_off >= 0:
            candidates.append((data, elf_off))

    if not candidates:
        raise RuntimeError(
            "No embedded ELF image found after gpu.binary"
        )

    # Normally there will be exactly one useful candidate.
    data, elf_off = max(
        candidates,
        key=lambda x: len(x[0])
    )

    try:
        elf_size = elf_file_size(data, elf_off)
        cubin = data[elf_off:elf_off + elf_size]
    except Exception as e:
        print(
            f"Warning: could not derive ELF extent ({e}); "
            "using bytes from ELF magic to end of container"
        )
        cubin = data[elf_off:]

    Path(args.output).write_bytes(cubin)

    print()
    print(f"ELF offset in container: {elf_off}")
    print(f"Wrote: {args.output}")
    print(f"Cubin bytes: {len(cubin)}")
    print(f"Magic: {cubin[:16].hex()}")


if __name__ == "__main__":
    main()
