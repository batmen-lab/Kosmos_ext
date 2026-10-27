from pathlib import Path
from itertools import islice
from autoevidence.execute.data_adapter import stage_dataset

root = Path(__file__).resolve().parents[1]
s = root / "data/myocardial fibrosis exp/staging/heart_eqtl_bulk_sym.csv"
o = Path("/tmp/heart_eqtl_check.csv")
n = 5

stage_dataset(str(s), o)

with s.open() as src, o.open() as out:
    src_head = list(islice(src, n + 1))
    out_head = list(islice(out, n + 1))

print(f"--- original: {s.name} (first {n} rows) ---")
print("".join(src_head), end="")
print(f"--- staged: {o.name} (first {n} rows) ---")
print("".join(out_head), end="")

with s.open() as src, o.open() as out:
    for i, (a, b) in enumerate(zip(src, out)):
        if "1.8301099999999999e-37" in a:
            print(f"--- line {i}, a value the old code changed to 1.83011e-37 ---")
            print("original:", a, end="")
            print("staged:  ", b, end="")
            break

print("byte-identical:", s.read_bytes() == o.read_bytes())
o.unlink()
