#!/usr/bin/env python3
"""Parse llama-eval-callback output into the refs.json shape validate expects:
[{name, op, dims, sum, head}] with head = first3+last3 of the first row."""
import json, re, sys

HEADER = re.compile(r"common_debug_cb_eval: +(.+?) = \((\w+)\) +(\S+)\(.*= \{([0-9, ]+)\}\s*$")
NUM = re.compile(r"-?(?:\d+\.\d+|inf|nan)")
SUM = re.compile(r"sum = (-?[\d.]+(?:e[+-]?\d+)?|nan|inf)")

records = []
cur = None
for line in open(sys.argv[1]):
    m = HEADER.search(line)
    if m:
        if cur:
            records.append(cur)
        dims = [int(x) for x in m.group(4).split(",")]
        cur = {"name": m.group(1).strip(), "op": m.group(3), "dims": dims,
               "sum": 0.0, "head": None}
        continue
    if cur is None:
        continue
    ms = SUM.search(line)
    if ms:
        try:
            cur["sum"] = float(ms.group(1))
        except ValueError:
            cur["sum"] = float("nan")
        continue
    if cur["head"] is None and "[" in line and any(c.isdigit() for c in line):
        vals = [float(v) for v in NUM.findall(line)]
        if len(vals) >= 3:
            cur["head"] = vals[:3] + vals[-3:]
if cur:
    records.append(cur)

# Trim trailing dims of 1 so dims[0] stays the row length.
for r in records:
    while len(r["dims"]) > 1 and r["dims"][-1] == 1:
        r["dims"].pop()

json.dump(records, open(sys.argv[2], "w"))
print(f"{len(records)} records -> {sys.argv[2]}", file=sys.stderr)
