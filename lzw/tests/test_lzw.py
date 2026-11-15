"""Run round trips and reject malformed LZW streams (pass the lzw executable)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(executable, *args, success=True):
    result = subprocess.run([str(executable), *map(str, args)], capture_output=True)
    if (result.returncode == 0) != success:
        raise AssertionError(
            f"command failed expectation: {args}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
    return result


def main():
    executable = Path(sys.argv[1]).resolve()
    block = 1 << 20
    cases = {
        "empty": b"",
        "single": b"A",
        "repeated": b"A" * (block + 17),
        "special_case": b"ABABABA" * 5000,
        "all_bytes": bytes(range(256)) * 200,
        "block_boundary": (b"ABCDE" * (block // 5 + 2))[: block + 3],
        "random_multiblock": os.urandom(block + 123),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, contents in cases.items():
            source = root / name
            packed = root / (name + ".lzw")
            restored = root / (name + ".restored")
            source.write_bytes(contents)
            run(executable, "v", source, 2)
            run(executable, "c", source, packed, 2)
            run(executable, "d", packed, restored, 2)
            assert restored.read_bytes() == contents, name
            if name == "single":
                damaged = root / "damaged.lzw"
                damaged.write_bytes(packed.read_bytes()[:-1])
                run(executable, "d", damaged, restored, 2, success=False)
                damaged.write_bytes(packed.read_bytes() + b"\0")
                run(executable, "d", damaged, restored, 2, success=False)
        print(f"OK: {len(cases)} round trips and malformed stream rejection")


if __name__ == "__main__":
    main()
