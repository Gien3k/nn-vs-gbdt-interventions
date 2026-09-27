"""Derive paper/main_blinded.tex from paper/main.tex (never edit the blinded
file by hand): the author block is replaced and self-identifying
declarations are neutralised.
"""
import re

from config import ROOT

PAPER = ROOT / "paper"


def main() -> None:
    s = (PAPER / "main.tex").read_text(encoding="utf-8")
    s, n = re.subn(r"%%AUTHORBLOCK-START.*?%%AUTHORBLOCK-END",
                   "\\\\author{\\\\fnm{Anonymous} \\\\sur{Author(s)}}\n"
                   "\\\\affil{\\\\orgname{Withheld for review}}",
                   s, flags=re.S)
    assert n == 1, "author block markers missing"
    s = re.sub(r"\\item \\textbf\{Author contribution:\}.*?(?=\\item|\\end\{itemize\})",
               "\\\\item \\\\textbf{Author contribution:} Withheld for review.\n",
               s, flags=re.S)
    # the public repository identifies the author: withhold it
    s = re.sub(r"%%REPO-URL|\\url\{https://github\.com/[^}]*\}"
               r"|\\url\{https://doi\.org/10\.5281/[^}]*\}",
               "[link withheld for review]", s)
    assert "Grabowski" not in s and "173145" not in s
    assert "github.com" not in s and "10.5281/zenodo" not in s
    (PAPER / "main_blinded.tex").write_text(s, encoding="utf-8")
    print("main_blinded.tex written")


if __name__ == "__main__":
    main()
