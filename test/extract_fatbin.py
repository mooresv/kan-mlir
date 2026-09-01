#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


FATBIN_MAGIC = b"\x50\xed\x55\xba"


def decode_mlir_string(s: str) -> bytes:
    out = bytearray()
    i = 0

    while i < len(s):
        if s[i] == "\\":
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
                f"unsupported escape at {i}: {s[i:i+12]!r}"
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

    fatbin = None

    for n, encoded in enumerate(quoted_strings(text, start)):
        try:
            data = decode_mlir_string(encoded)
        except RuntimeError:
            continue

        if len(data) > 16:
            print(
                f"quoted[{n}]: "
                f"decoded={len(data)} "
                f"magic={data[:8].hex()}"
            )

        if data.startswith(FATBIN_MAGIC):
            fatbin = data
            break

    if fatbin is None:
        raise RuntimeError(
            "could not find NVIDIA fatbinary payload"
        )

    Path(args.output).write_bytes(fatbin)

    print()
    print(f"Wrote: {args.output}")
    print(f"Fatbin bytes: {len(fatbin)}")
    print(f"Magic: {fatbin[:16].hex()}")


if __name__ == "__main__":
    main()
