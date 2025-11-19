# Rhadamanthys loader deobfuscator

This is a companion repo to the blog post [Rhadamanthys loader deobfuscation](https://cyber.wtf/2025/11/19/rhadamanthys-loader-deobfuscation/).

The purpose of the individual scripts should be self-explanatory.
For a quick start, run `deob.py` against your obfuscated binary.
You can optionally pass a second parameter for the output filename, otherwise `.patched` will be appended.

Terminology used in the scripts:
* `va`: Virtual address.
* `site`: Obfuscated place of interest currently being worked on.
* `extent`: Contiguous stretch of virtual addresses with a size > 0.
* `slice`: Collection of filtered instructions (e.g., all that contribute to the value of a register.)
* `anchor`: Existing (usually obsolete) instruction that is used as a placement hint for new instruction(s).
