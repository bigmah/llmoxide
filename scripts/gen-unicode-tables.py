#!/usr/bin/env python3
"""Emit Rust unicode tables from llama.cpp's unicode-data.cpp.

Extracts contiguous codepoint ranges for \p{L}, \p{M}, \p{N} and the
whitespace set, so llmoxide's pretokenizer classifies exactly like llama.cpp.
"""
import re, sys

src = open(sys.argv[1]).read()

# unicode_ranges_flags: {start, flags} pairs; each range runs to next start-1.
m = re.search(r"unicode_ranges_flags = \{(.*?)\n\};", src, re.S)
pairs = re.findall(r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}", m.group(1))
pairs = [(int(a, 16), int(b, 16)) for a, b in pairs]

NUMBER, LETTER, MARK = 0x0002, 0x0004, 0x0010
MAX_CPT = 0x110000

def class_of(flags):
    if flags & LETTER: return "L"
    if flags & MARK:   return "M"
    if flags & NUMBER: return "N"
    return None

ranges = {"L": [], "M": [], "N": []}
for i, (start, flags) in enumerate(pairs):
    end = (pairs[i + 1][0] if i + 1 < len(pairs) else MAX_CPT) - 1
    c = class_of(flags)
    if c is None:
        continue
    if ranges[c] and ranges[c][-1][1] == start - 1:
        ranges[c][-1] = (ranges[c][-1][0], end)
    else:
        ranges[c].append((start, end))

m = re.search(r"unicode_set_whitespace = \{(.*?)\n\};", src, re.S)
ws = sorted(int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]+)", m.group(1)))
ws_ranges = []
for c in ws:
    if ws_ranges and ws_ranges[-1][1] == c - 1:
        ws_ranges[-1] = (ws_ranges[-1][0], c)
    else:
        ws_ranges.append((c, c))

def emit(name, rs):
    print(f"pub const {name}: &[(u32, u32)] = &[")
    for i in range(0, len(rs), 6):
        row = ", ".join(f"(0x{a:X}, 0x{b:X})" for a, b in rs[i:i + 6])
        print(f"    {row},")
    print("];\n")

print("//! Unicode category ranges, generated from llama.cpp b10090's")
print("//! `unicode-data.cpp` (scripts/gen-unicode-data.py output) so that the")
print("//! pretokenizer classifies codepoints exactly like the reference")
print("//! implementation. Half-open is not used: ranges are inclusive.")
print()
emit("LETTER", ranges["L"])
emit("MARK", ranges["M"])
emit("NUMBER", ranges["N"])
emit("WHITESPACE", ws_ranges)
print(f"// {len(ranges['L'])} letter, {len(ranges['M'])} mark, {len(ranges['N'])} number, {len(ws_ranges)} whitespace ranges", file=sys.stderr)
