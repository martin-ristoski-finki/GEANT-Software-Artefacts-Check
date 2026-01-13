from __future__ import annotations
import sys
import subprocess
import tempfile
import csv
import re
from pathlib import Path

ARTEFACTS = {
    "README": ["README", "README.md", "README.rst", "README.txt"],
    "LICENSE": ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt"],
    "NOTICE": ["NOTICE", "NOTICE.txt", "NOTICE.md"],
    "COPYRIGHT": ["COPYRIGHT", "COPYRIGHT.txt", "COPYRIGHT.md"],
    "AUTHORS": ["AUTHORS", "AUTHORS.txt", "AUTHORS.md"],
    "CHANGELOG": ["CHANGELOG", "CHANGELOG.md", "CHANGELOG.txt", "Changes.md", "CHANGES", "CHANGES.md"],
    "CONTRIBUTING": ["CONTRIBUTING", "CONTRIBUTING.md", "CONTRIBUTING.txt"],
}

# Next checks: binaries / archives / compiled artefacts
BINARY_EXTENSIONS = {
    ".apk", ".exe", ".msi", ".dmg", ".jar", ".war", ".dll", ".so", ".dylib",
    ".bin", ".iso", ".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".o", ".obj", ".class", ".pyc"
}

# Large file threshold (bytes) – start conservative
LARGE_FILE_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10 MB

# directories to ignore during scans (optional; keep small to avoid missing issues)
IGNORE_DIRS = {".git", ".github", ".idea", ".vscode", "__pycache__", ".venv", "venv", "node_modules"}

def run(cmd: list[str], cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")

def find_any(repo_root: Path, names: list[str]) -> list[str]:
    found = []
    for n in names:
        p = repo_root / n
        if p.exists() and p.is_file():
            found.append(n)
    return found

def iter_files(repo_root: Path):
    """Yield files under repo_root, skipping IGNORE_DIRS."""
    for p in repo_root.rglob("*"):
        if p.is_dir():
            continue
        # Skip ignored directories
        parts = set(p.parts)
        if parts.intersection(IGNORE_DIRS):
            continue
        yield p

def scan_binaries(repo_root: Path) -> list[dict]:
    """Return rows for files matching BINARY_EXTENSIONS."""
    rows = []
    for f in iter_files(repo_root):
        ext = f.suffix.lower()
        if ext in BINARY_EXTENSIONS:
            size = f.stat().st_size
            rows.append({
                "check_type": "binary_file",
                "path": str(f.relative_to(repo_root)).replace("\\", "/"),
                "size_bytes": str(size),
                "reason": f"Extension {ext} flagged"
            })
    return rows

def scan_large_files(repo_root: Path, threshold: int) -> list[dict]:
    """Return rows for files exceeding size threshold."""
    rows = []
    for f in iter_files(repo_root):
        size = f.stat().st_size
        if size >= threshold:
            rows.append({
                "check_type": "large_file",
                "path": str(f.relative_to(repo_root)).replace("\\", "/"),
                "size_bytes": str(size),
                "reason": f"File size >= {threshold} bytes"
            })
    return rows

def read_text_safe(path: Path, max_bytes: int = 500_000) -> str:
    """
    Read text safely; limits size to avoid huge files.
    Tries utf-8 then latin-1 as fallback.
    """
    data = path.read_bytes()[:max_bytes]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode("latin-1", errors="replace")

def notice_heuristics(notice_path: Path, repo_name: str) -> list[dict]:
    """
    Basic, non-legal heuristics for NOTICE completeness.
    Produces PASS/WARN/FAIL rows for the user's NOTICE checklist items.
    """
    text = read_text_safe(notice_path).lower()

    def has(pattern: str) -> bool:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    results = []

    # 1) Project name
    # Heuristic: repo name appears OR "project" header exists
    proj_ok = (repo_name.lower() in text) or has(r"^\s*project\s*[:\-]",)
    results.append({
        "check_type": "notice_content",
        "item": "Project name",
        "status": "PASS" if proj_ok else "WARN",
        "comment": "Project name detected" if proj_ok else "Project name not clearly detected in NOTICE"
    })

    # 2) Copyright one-liner
    c_ok = has(r"copyright") and has(r"(19|20)\d{2}")
    results.append({
        "check_type": "notice_content",
        "item": "Copyright – short one-liner",
        "status": "PASS" if c_ok else "FAIL",
        "comment": "Copyright line/year detected" if c_ok else "No clear copyright line with year detected"
    })

    # 3) Licence declaration/link
    # Heuristic: mentions license/licence and has URL or references LICENSE/COPYING
    lic_ok = has(r"licen[cs]e") and (has(r"https?://") or has(r"\blicense\b") or has(r"\bcopying\b"))
    results.append({
        "check_type": "notice_content",
        "item": "Licence – declaration/link to full text",
        "status": "PASS" if lic_ok else "WARN",
        "comment": "Licence mention and reference detected" if lic_ok else "Licence mention/reference not clearly detected"
    })

    # 4) Authors link
    auth_ok = has(r"\bauthors?\b") or has(r"\bcontributors?\b") or has(r"\bsee\s+authors\b") or has(r"\bauthors\.") or has(r"\bAUTHORS\b".lower())
    results.append({
        "check_type": "notice_content",
        "item": "Authors and contributors – or link to AUTHORS",
        "status": "PASS" if auth_ok else "WARN",
        "comment": "Authors/contributors reference detected" if auth_ok else "No clear authors/contributors reference detected"
    })

    # 5) Third-party components
    # Heuristic: words like "third party" OR multiple occurrences of "license:" / "version"
    third_ok = has(r"third[-\s]?party") or (text.count("license") + text.count("licence") >= 2) or (text.count("version") >= 2)
    results.append({
        "check_type": "notice_content",
        "item": "Third-party components – name/version/licence",
        "status": "PASS" if third_ok else "WARN",
        "comment": "Third-party section/list indicators detected" if third_ok else "No clear third-party component listing detected"
    })

    # 6) Tools used
    tools_ok = has(r"\btools?\b") or has(r"\bbuilt with\b") or has(r"\buses\b.*\btool\b")
    results.append({
        "check_type": "notice_content",
        "item": "Tools used – name/link/purpose",
        "status": "WARN" if tools_ok else "WARN",
        "comment": "Tools section detection is heuristic; manual confirmation recommended"
    })

    # 7) Trademark disclaimer
    tm_ok = has(r"trademark") or ("™" in text) or ("®" in text) or has(r"not affiliated")
    results.append({
        "check_type": "notice_content",
        "item": "Trademark disclaimer",
        "status": "PASS" if tm_ok else "WARN",
        "comment": "Trademark wording detected" if tm_ok else "No trademark wording detected"
    })

    # 8) Patents statements
    pat_ok = has(r"patent")
    results.append({
        "check_type": "notice_content",
        "item": "Patents and patent-related statements",
        "status": "PASS" if pat_ok else "WARN",
        "comment": "Patent wording detected" if pat_ok else "No patent wording detected"
    })

    # 9) Special acknowledgements
    ack_ok = has(r"acknowledg") or has(r"thanks") or has(r"funding") or has(r"funded by")
    results.append({
        "check_type": "notice_content",
        "item": "Special acknowledgements or attributions",
        "status": "PASS" if ack_ok else "WARN",
        "comment": "Acknowledgements/funding wording detected" if ack_ok else "No acknowledgements wording detected"
    })

    # 10) Prior notices included
    prior_ok = has(r"prior notice") or has(r"this product includes") or has(r"third[-\s]?party notices")
    results.append({
        "check_type": "notice_content",
        "item": "Prior notices for third-party components when required",
        "status": "PASS" if prior_ok else "WARN",
        "comment": "Prior-notice wording detected" if prior_ok else "No explicit prior-notice wording detected"
    })

    return results

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python check_repo_artefacts.py https://github.com/GEANT/CAT.git")
        return 2

    git_url = sys.argv[1].strip()
    csv_file = Path("artefact_check.csv")

    rows = []

    with tempfile.TemporaryDirectory(prefix="artefact_check_") as tmpdir:
        repo_dir = Path(tmpdir) / "repo"

        print(f"Cloning: {git_url}")
        run(["git", "clone", "--depth", "1", git_url, str(repo_dir)])

        if not (repo_dir / ".git").exists():
            print("ERROR: Clone did not produce a git repository.")
            return 2

        repo_name = repo_dir.name  # fallback; we can derive from URL too if you want

        # ---- 1) Key artefacts presence (your original check)
        for artefact, candidates in ARTEFACTS.items():
            found = find_any(repo_dir, candidates)
            status = "PASS" if found else "FAIL"
            rows.append({
                "repo_url": git_url,
                "check_type": "key_artefact",
                "item": artefact,
                "status": status,
                "details": ", ".join(found)
            })

        # ---- 2) Binary artefacts check
        bin_rows = scan_binaries(repo_dir)
        if not bin_rows:
            rows.append({
                "repo_url": git_url,
                "check_type": "binary_file",
                "item": "Binary artefacts",
                "status": "PASS",
                "details": "No flagged binary/compiled/archive files found"
            })
        else:
            for r in bin_rows:
                rows.append({
                    "repo_url": git_url,
                    "check_type": r["check_type"],
                    "item": r["path"],
                    "status": "FAIL",
                    "details": f'{r["reason"]}; size={r["size_bytes"]}'
                })

        # ---- 3) Large files check
        large_rows = scan_large_files(repo_dir, LARGE_FILE_THRESHOLD_BYTES)
        if not large_rows:
            rows.append({
                "repo_url": git_url,
                "check_type": "large_file",
                "item": f"Large files (>= {LARGE_FILE_THRESHOLD_BYTES} bytes)",
                "status": "PASS",
                "details": "No large files found above threshold"
            })
        else:
            for r in large_rows:
                rows.append({
                    "repo_url": git_url,
                    "check_type": r["check_type"],
                    "item": r["path"],
                    "status": "WARN",
                    "details": f'{r["reason"]}; size={r["size_bytes"]}'
                })

        # ---- 4) NOTICE content heuristics (only if NOTICE exists)
        notice_found = find_any(repo_dir, ARTEFACTS["NOTICE"])
        if notice_found:
            notice_path = repo_dir / notice_found[0]
            for r in notice_heuristics(notice_path, repo_name=repo_name):
                rows.append({
                    "repo_url": git_url,
                    "check_type": r["check_type"],
                    "item": r["item"],
                    "status": r["status"],
                    "details": r["comment"]
                })
        else:
            # If NOTICE missing, we already FAIL it above in key artefacts.
            # Add a note that content checks were skipped.
            rows.append({
                "repo_url": git_url,
                "check_type": "notice_content",
                "item": "NOTICE content checks",
                "status": "SKIP",
                "details": "NOTICE file not found; content heuristics skipped"
            })

    # Write CSV (combined)
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["repo_url", "check_type", "item", "status", "details"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV report written to: {csv_file.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
