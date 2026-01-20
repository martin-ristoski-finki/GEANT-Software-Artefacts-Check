from __future__ import annotations
import sys
import subprocess
import tempfile
import csv
import re
from pathlib import Path
from typing import Optional, Iterable

# ============================================================================
# CONFIGURATION
# ============================================================================

ARTEFACTS = {
    "README": ["README", "README.md", "README.rst", "README.txt"],
    "LICENSE": ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt"],
    "NOTICE": ["NOTICE", "NOTICE.txt", "NOTICE.md"],
    "COPYRIGHT": ["COPYRIGHT", "COPYRIGHT.txt", "COPYRIGHT.md"],
    "AUTHORS": ["AUTHORS", "AUTHORS.txt", "AUTHORS.md"],
    "CHANGELOG": ["CHANGELOG", "CHANGELOG.md", "CHANGELOG.txt", "Changes.md", "CHANGES", "CHANGES.md"],
    "CONTRIBUTING": ["CONTRIBUTING", "CONTRIBUTING.md", "CONTRIBUTING.txt"],
}

BINARY_EXTENSIONS = {
    ".apk", ".exe", ".msi", ".dmg", ".jar", ".war", ".dll", ".so", ".dylib",
    ".bin", ".iso", ".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".o", ".obj", ".class", ".pyc"
}

LARGE_FILE_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10 MB
IGNORE_DIRS = {".git", ".github", ".idea", ".vscode", "__pycache__", ".venv", "venv", "node_modules"}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Execute git command and raise on failure."""
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")


def find_any(repo_root: Path, names: list[str]) -> list[str]:
    """Find any file matching the given names in repo root."""
    found = []
    for n in names:
        p = repo_root / n
        if p.exists() and p.is_file():
            found.append(n)
    return found


def repo_name_from_url(url: str) -> str:
    """Extract repository name from git URL."""
    tail = url.rstrip("/").split("/")[-1]
    return tail[:-4] if tail.endswith(".git") else tail


def iter_files(repo_root: Path) -> Iterable[Path]:
    """Yield files under repo_root, skipping IGNORE_DIRS."""
    for p in repo_root.rglob("*"):
        if p.is_dir():
            continue
        rel_parts = set(p.relative_to(repo_root).parts)
        if rel_parts.intersection(IGNORE_DIRS):
            continue
        yield p


def read_text_safe(path: Path, max_bytes: int = 800_000) -> str:
    """Read text file safely with encoding fallback."""
    data = path.read_bytes()[:max_bytes]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode("latin-1", errors="replace")


def normalize(s: str) -> str:
    """Normalize whitespace in string."""
    return re.sub(r"\s+", " ", s.strip())


def has_any(text_lc: str, patterns: list[str]) -> bool:
    """Check if any regex pattern matches text."""
    return any(re.search(p, text_lc, flags=re.IGNORECASE) for p in patterns)


def add_row(rows: list[dict], git_url: str, check_type: str, item: str, status: str, details: str) -> None:
    """Add a row to results list."""
    rows.append({
        "repo_url": git_url,
        "check_type": check_type,
        "item": item,
        "status": status,
        "details": details
    })


# ============================================================================
# BINARY & LARGE FILE CHECKS
# ============================================================================

def scan_binaries(repo_root: Path) -> list[dict]:
    """Find binary/compiled files."""
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
    """Find files exceeding size threshold."""
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


# ============================================================================
# README CHECKS
# ============================================================================

MD_BADGE_RE = re.compile(r"!\[[^\]]*\]\([^\)]+\)", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def extract_headings(md: str) -> list[str]:
    """Extract Markdown headings."""
    return [normalize(m.group(2)) for m in HEADING_RE.finditer(md)]


def readme_path(repo_root: Path) -> Optional[Path]:
    """Find README file, preferring .md extension."""
    found = find_any(repo_root, ARTEFACTS["README"])
    if not found:
        return None
    for pref in ["README.md", "README.rst", "README.txt", "README"]:
        if pref in found:
            return repo_root / pref
    return repo_root / found[0]


def local_links_in_readme(md: str) -> list[str]:
    """Extract local file links from README."""
    links = []
    for _, target in MD_LINK_RE.findall(md):
        target = target.strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if target:
            links.append(target)
    return links


def readme_checks(repo_root: Path, git_url: str, repo_name: str) -> list[dict]:
    """Comprehensive README content checks."""
    rows: list[dict] = []
    rp = readme_path(repo_root)

    if rp is None:
        add_row(rows, git_url, "readme_check", "README content checks", "SKIP", "README not found")
        return rows

    md = read_text_safe(rp)
    text_lc = md.lower()
    headings = [h.lower() for h in extract_headings(md)]
    repo_name_lc = repo_name.lower()

    # Project name
    title_ok = (repo_name_lc in text_lc[:4000]) or any(repo_name_lc in h for h in headings)
    add_row(rows, git_url, "readme_check", "Project name",
            "PASS" if title_ok else "WARN",
            "Detected repo/project name" if title_ok else "Project name not clearly detected")

    # Copyright one-liner
    cr_ok = bool(re.search(r"copyright\s*(?:\(|©)?\s*(19|20)\d{2}", text_lc))
    add_row(rows, git_url, "readme_check", "Copyright – short one-liner",
            "PASS" if cr_ok else "WARN",
            "Copyright + year detected" if cr_ok else "No clear copyright one-liner detected")

    # Project status
    status_ok = has_any(text_lc, [r"\bstatus\b", r"\blifecycle\b", r"\bbeta\b", r"\bstable\b",
                                  r"\bretired\b", r"\bdeprecated\b", r"\bdevelopment\b"])
    add_row(rows, git_url, "readme_check", "Project status – software lifecycle stage",
            "PASS" if status_ok else "WARN",
            "Lifecycle/status keywords detected" if status_ok else "No explicit lifecycle/status detected")

    # Tags
    tags_ok = has_any(text_lc, [r"^\s*tags?\s*[:\-]", r"\bkeywords?\b"])
    add_row(rows, git_url, "readme_check", "Tags – relevant tags and categories",
            "PASS" if tags_ok else "WARN",
            "Tags/keywords detected" if tags_ok else "No tags/keywords detected")

    # Version
    version_ok = has_any(text_lc,
                         [r"\bversion\s*v?\d+\.\d+(\.\d+)?", r"\brelease\s*v?\d+\.\d+", r"\bcurrent\s+version\b"])
    add_row(rows, git_url, "readme_check", "Latest stable version indicated",
            "PASS" if version_ok else "WARN",
            "Version/release pattern detected" if version_ok else "No clear version indication")

    # Licence one-liner
    lic_ok = has_any(text_lc, [r"\blicen[cs]ed\s+under\b", r"\blicen[cs]e\s*[:\-]",
                               r"\bMIT\b|\bApache\b|\bGPL\b|\bEUPL\b|\bBSD\b"]) and \
             has_any(text_lc, [r"\bLICENSE\b", r"\bCOPYING\b", r"https?://"])
    add_row(rows, git_url, "readme_check", "Licence – short one-liner with link",
            "PASS" if lic_ok else "WARN",
            "Licence mention + reference detected" if lic_ok else "Licence one-liner/link not detected")

    # Badges
    badge_count = len(MD_BADGE_RE.findall(md))
    add_row(rows, git_url, "readme_check", "Badges",
            "PASS" if badge_count > 0 else "WARN",
            f"Found {badge_count} badge(s)" if badge_count > 0 else "No badges detected")

    # Badge categories
    badge_categories = [
        ("Badges: Project metadata and release status", [r"release", r"tag", r"version", r"latest"]),
        ("Badges: Package downloads and versions", [r"pypi", r"npm", r"maven", r"nuget", r"downloads"]),
        ("Badges: Community engagement", [r"contributors", r"contributing", r"prs\s+welcome"]),
        ("Badges: CI/CD workflows", [r"github\.com/.*/actions", r"travis", r"circleci", r"build\s+status"]),
        ("Badges: Code quality", [r"codeql", r"sonar", r"codeclimate", r"quality"]),
        ("Badges: Test coverage", [r"codecov", r"coveralls", r"coverage"]),
        ("Badges: Security vulnerabilities", [r"snyk", r"dependabot", r"vulnerab", r"security"]),
        ("Badges: Documentation", [r"readthedocs", r"docs", r"documentation\s+status"]),
        ("Badges: Community support", [r"slack", r"discord", r"discourse", r"gitter"]),
    ]

    for name, pats in badge_categories:
        ok = any(re.search(p, md, flags=re.IGNORECASE) for p in pats)
        add_row(rows, git_url, "readme_check", name,
                "PASS" if ok else "WARN",
                "Detected" if ok else "Not detected")

    # Description
    desc_ok = len(text_lc) >= 100 and not text_lc.strip().startswith("#")
    add_row(rows, git_url, "readme_check", "Description – clear and concise overview",
            "PASS" if desc_ok else "WARN",
            "Introduction content detected" if desc_ok else "No clear description/overview")

    # Features
    feat_ok = ("features" in headings) or has_any(text_lc, [r"^\s*features?\s*[:\-]", r"## features"])
    add_row(rows, git_url, "readme_check", "Features – key functionalities",
            "PASS" if feat_ok else "WARN",
            "Features section detected" if feat_ok else "No features section")

    # Scope
    scope_ok = any(h in headings for h in ["scope", "use cases", "limitations", "constraints"]) or \
               has_any(text_lc, [r"\bscope\b", r"\buse\s+case", r"\blimitations?\b"])
    add_row(rows, git_url, "readme_check", "Scope – contexts, use cases, limitations",
            "PASS" if scope_ok else "WARN",
            "Scope/use-case signals detected" if scope_ok else "No clear scope section")

    # Supported environments
    supp_ok = any(h in headings for h in ["supported", "compatibility", "platforms"]) or \
              has_any(text_lc, [r"\bsupported\b", r"\bcompatib", r"\bplatforms?\b"])
    add_row(rows, git_url, "readme_check", "Supported tools/environments/clients",
            "PASS" if supp_ok else "WARN",
            "Compatibility signals detected" if supp_ok else "No compatibility section")

    # Components/structure
    comp_ok = any(h in headings for h in ["architecture", "components", "structure"]) or \
              has_any(text_lc, [r"\barchitecture\b", r"\bcomponents?\b", r"\bproject\s+structure\b"])
    add_row(rows, git_url, "readme_check", "Components and/or project structure",
            "PASS" if comp_ok else "WARN",
            "Structure signals detected" if comp_ok else "No structure section")

    # System requirements
    req_ok = any(h in headings for h in ["requirements", "prerequisites"]) or \
             has_any(text_lc, [r"\brequirements?\b", r"\bprerequisites?\b"])
    add_row(rows, git_url, "readme_check", "System requirements",
            "PASS" if req_ok else "WARN",
            "Requirements detected" if req_ok else "No requirements section")

    # Installation
    inst_ok = any(h in headings for h in ["installation", "install", "setup", "getting started"]) or \
              has_any(text_lc, [r"\binstall", r"\bsetup\b", r"\bgetting started\b"])
    add_row(rows, git_url, "readme_check", "Installation – instructions or link",
            "PASS" if inst_ok else "WARN",
            "Installation signals detected" if inst_ok else "No installation section")

    # Usage
    usage_ok = any(h in headings for h in ["usage", "how to use", "examples", "quickstart"]) or \
               has_any(text_lc, [r"\busage\b", r"\bexample", r"\bquickstart\b"])
    codeblock_ok = "```" in md
    add_row(rows, git_url, "readme_check", "Usage – examples/demos",
            "PASS" if (usage_ok and codeblock_ok) else "WARN",
            "Usage + code blocks detected" if (usage_ok and codeblock_ok) else "Usage/examples not clearly present")

    # Documentation
    docs_ok = any(h in headings for h in ["documentation", "docs"]) or \
              has_any(text_lc, [r"\bdocumentation\b", r"\bdocs/\b", r"readthedocs"])
    add_row(rows, git_url, "readme_check", "Documentation – location",
            "PASS" if docs_ok else "WARN",
            "Documentation signals detected" if docs_ok else "No documentation location")

    # Version control
    vc_ok = has_any(text_lc, [r"\bbranch", r"\btag(s|ging)?", r"\brelease"])
    add_row(rows, git_url, "readme_check", "Version control – branch structure/tagging",
            "PASS" if vc_ok else "WARN",
            "Branch/tag wording detected" if vc_ok else "No branch/tagging conventions")

    # Troubleshooting/FAQ
    faq_ok = any(h in headings for h in ["troubleshooting", "faq"]) or \
             has_any(text_lc, [r"\btroubleshoot", r"\bfaq\b"])
    add_row(rows, git_url, "readme_check", "Troubleshooting & FAQ",
            "PASS" if faq_ok else "WARN",
            "Troubleshooting/FAQ detected" if faq_ok else "No troubleshooting/FAQ")

    # Support
    support_ok = has_any(text_lc, [r"\bissues\b", r"\bsupport\b", r"\bcontact\b"])
    add_row(rows, git_url, "readme_check", "Support – contact/issue tracker",
            "PASS" if support_ok else "WARN",
            "Support/contact wording detected" if support_ok else "No support info")

    # Privacy policy
    privacy_ok = has_any(text_lc, [r"\bprivacy\b", r"privacy policy"])
    add_row(rows, git_url, "readme_check", "Privacy policy (if applicable)",
            "PASS" if privacy_ok else "WARN",
            "Privacy wording detected" if privacy_ok else "No privacy policy")

    # Roadmap
    roadmap_ok = any(h in headings for h in ["roadmap"]) or \
                 has_any(text_lc, [r"\broadmap\b", r"\bplanned\b", r"\bfuture\b"])
    add_row(rows, git_url, "readme_check", "Roadmap",
            "PASS" if roadmap_ok else "WARN",
            "Roadmap detected" if roadmap_ok else "No roadmap")

    # Authors mention
    authors_ok = has_any(text_lc, [r"\bauthors?\b", r"\bcontributors?\b"])
    add_row(rows, git_url, "readme_check", "Authors – mention/link",
            "PASS" if authors_ok else "WARN",
            "Authors mentioned" if authors_ok else "No authors mention")

    # Contributing
    contrib_ok = any(h in headings for h in ["contributing"]) or \
                 has_any(text_lc, [r"\bcontributing\b", r"\bhow to contribute\b"])
    add_row(rows, git_url, "readme_check", "Contributing – guidelines/link",
            "PASS" if contrib_ok else "WARN",
            "Contributing detected" if contrib_ok else "No contributing guidance")

    # Funding
    fund_ok = has_any(text_lc, [r"\bfunding\b", r"\bgrant\b", r"\bhorizon\b", r"\beu\b"])
    add_row(rows, git_url, "readme_check", "Funding – sources/grant",
            "PASS" if fund_ok else "WARN",
            "Funding wording detected" if fund_ok else "No funding info")

    # Acknowledgements
    ack_ok = has_any(text_lc, [r"\backnowledg", r"\bthanks\b"])
    add_row(rows, git_url, "readme_check", "Other acknowledgements",
            "PASS" if ack_ok else "WARN",
            "Acknowledgements detected" if ack_ok else "No acknowledgements")

    # Dependencies
    dep_ok = has_any(text_lc, [r"\bdependencies\b", r"\brequirements\b"]) or \
             any((repo_root / f).exists() for f in
                 ["package.json", "requirements.txt", "pyproject.toml", "pom.xml", "go.mod"])
    add_row(rows, git_url, "readme_check", "Dependencies – main items listed",
            "PASS" if dep_ok else "WARN",
            "Dependencies/manifests detected" if dep_ok else "No dependencies section")

    # Tools used
    tools_ok = has_any(text_lc, [r"\btools?\b", r"\bbuilt with\b"]) or \
               any((repo_root / f).exists() for f in ["Makefile", "Dockerfile"])
    add_row(rows, git_url, "readme_check", "Tools used – development/build",
            "PASS" if tools_ok else "WARN",
            "Tools indicators detected" if tools_ok else "No tools indicators")

    # Licence paragraph
    lic_section = "license" in headings or has_any(text_lc, [r"^#+\s*licen[cs]e\b"])
    lic_para_ok = lic_section and has_any(text_lc, [r"\bLICENSE\b", r"https?://"])
    add_row(rows, git_url, "readme_check", "Licence – summary paragraph + link",
            "PASS" if lic_para_ok else "WARN",
            "Licence section detected" if lic_para_ok else "No licence summary")

    # Copyright details
    cd_ok = has_any(text_lc, [r"\bCOPYRIGHT\b", r"copyright\s*(19|20)\d{2}"])
    add_row(rows, git_url, "readme_check", "Copyright – additional details/link",
            "PASS" if cd_ok else "WARN",
            "COPYRIGHT reference detected" if cd_ok else "No COPYRIGHT reference")

    # Validate local links
    links = local_links_in_readme(md)
    broken = []
    for t in links:
        candidate = (repo_root / t).resolve()
        if str(candidate).startswith(str(repo_root.resolve())) and not candidate.exists():
            broken.append(t)
    add_row(rows, git_url, "readme_check", "README local links resolve",
            "PASS" if not broken else "WARN",
            "All links valid" if not broken else f"Broken links: {', '.join(broken[:5])}")

    return rows


# ============================================================================
# LICENSE CHECKS
# ============================================================================

def license_checks(repo_root: Path, git_url: str) -> list[dict]:
    """LICENSE file content checks."""
    rows: list[dict] = []
    found = find_any(repo_root, ARTEFACTS["LICENSE"])

    if not found:
        add_row(rows, git_url, "license_check", "LICENSE content checks", "SKIP", "LICENSE not found")
        return rows

    lic_path = repo_root / found[0]
    text = read_text_safe(lic_path)
    text_lc = text.lower()

    # Full official license text (heuristic: substantial content)
    full_text_ok = len(text) >= 500
    add_row(rows, git_url, "license_check", "Full official licence text",
            "PASS" if full_text_ok else "WARN",
            f"LICENSE file has {len(text)} characters" if full_text_ok else "LICENSE file seems too short")

    # Common license detection
    licenses_detected = []
    license_patterns = {
        "MIT": r"\bMIT License\b",
        "Apache": r"\bApache License",
        "GPL": r"\bGNU General Public License",
        "BSD": r"\bBSD License",
        "EUPL": r"\bEuropean Union Public Licence",
    }

    for lic_name, pattern in license_patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            licenses_detected.append(lic_name)

    add_row(rows, git_url, "license_check", "License type detection",
            "PASS" if licenses_detected else "WARN",
            f"Detected: {', '.join(licenses_detected)}" if licenses_detected else "No standard license detected")

    return rows


# ============================================================================
# COPYRIGHT CHECKS
# ============================================================================

def copyright_checks(repo_root: Path, git_url: str) -> list[dict]:
    """COPYRIGHT file content checks."""
    rows: list[dict] = []
    found = find_any(repo_root, ARTEFACTS["COPYRIGHT"])

    if not found:
        add_row(rows, git_url, "copyright_check", "COPYRIGHT content checks", "SKIP", "COPYRIGHT not found")
        return rows

    cr_path = repo_root / found[0]
    text = read_text_safe(cr_path)
    text_lc = text.lower()

    # Copyright statement with year
    cr_year_ok = bool(re.search(r"copyright\s*(?:\(|©)?\s*(19|20)\d{2}", text_lc))
    add_row(rows, git_url, "copyright_check", "Copyright statement with years",
            "PASS" if cr_year_ok else "WARN",
            "Copyright + year detected" if cr_year_ok else "No clear copyright statement with year")

    # GÉANT mention (optional for non-GÉANT projects)
    geant_ok = "geant" in text_lc or "géant" in text_lc
    add_row(rows, git_url, "copyright_check", "GÉANT copyright (if applicable)",
            "PASS" if geant_ok else "WARN",
            "GÉANT mentioned" if geant_ok else "No GÉANT mention (may not be required)")

    # Multiple copyright holders
    copyright_count = len(re.findall(r"copyright\s*(?:\(|©)?", text_lc))
    add_row(rows, git_url, "copyright_check", "Copyright statements present",
            "PASS" if copyright_count >= 1 else "WARN",
            f"Found {copyright_count} copyright statement(s)")

    # EU logo check (file-based)
    eu_logo_ok = any((repo_root / f).exists() for f in ["eu-logo.png", "eu-flag.png", "EU-logo.jpg", "EU-flag.jpg"])
    add_row(rows, git_url, "copyright_check", "EU logo present (if required)",
            "PASS" if eu_logo_ok else "WARN",
            "EU logo file detected" if eu_logo_ok else "No EU logo file detected")

    return rows


# ============================================================================
# AUTHORS CHECKS
# ============================================================================

def authors_checks(repo_root: Path, git_url: str) -> list[dict]:
    """AUTHORS file content checks."""
    rows: list[dict] = []
    found = find_any(repo_root, ARTEFACTS["AUTHORS"])

    if not found:
        add_row(rows, git_url, "authors_check", "AUTHORS content checks", "SKIP", "AUTHORS not found")
        return rows

    auth_path = repo_root / found[0]
    text = read_text_safe(auth_path)
    text_lc = text.lower()

    # GÉANT phase/work package
    geant_phase_ok = bool(re.search(r"gn\d+|géant\s+\d+|work\s+package|wp\d+|task\s+\d+", text_lc))
    add_row(rows, git_url, "authors_check", "GÉANT phase/work package (if applicable)",
            "PASS" if geant_phase_ok else "WARN",
            "GÉANT phase/WP detected" if geant_phase_ok else "No GÉANT phase/WP detected")

    # Developers list (heuristic: multiple names/emails)
    email_count = len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text))
    name_lines = len([line for line in text.split("\n") if line.strip() and not line.strip().startswith("#")])
    add_row(rows, git_url, "authors_check", "Developers with contacts",
            "PASS" if email_count >= 1 or name_lines >= 2 else "WARN",
            f"Found {email_count} email(s), {name_lines} non-empty lines")

    # Contributors mention
    contrib_ok = has_any(text_lc, [r"\bcontributors?\b", r"\bother\b"])
    add_row(rows, git_url, "authors_check", "Other contributors mentioned",
            "PASS" if contrib_ok else "WARN",
            "Contributors section detected" if contrib_ok else "No contributors section")

    # Funding information
    fund_ok = has_any(text_lc, [r"\bfunding\b", r"\bgrant\b", r"\bsponsored\b"])
    add_row(rows, git_url, "authors_check", "Funding information",
            "PASS" if fund_ok else "WARN",
            "Funding info detected" if fund_ok else "No funding info")

    return rows


# ============================================================================
# NOTICE CHECKS
# ============================================================================

def notice_checks(repo_root: Path, git_url: str, repo_name: str) -> list[dict]:
    """NOTICE file content checks."""
    rows: list[dict] = []
    found = find_any(repo_root, ARTEFACTS["NOTICE"])

    if not found:
        add_row(rows, git_url, "notice_check", "NOTICE content checks", "SKIP", "NOTICE not found")
        return rows

    notice_path = repo_root / found[0]
    text = read_text_safe(notice_path)
    text_lc = text.lower()

    # Project name
    proj_ok = (repo_name.lower() in text_lc) or has_any(text_lc, [r"^\s*project\s*[:\-]"])
    add_row(rows, git_url, "notice_check", "Project name",
            "PASS" if proj_ok else "WARN",
            "Project name detected" if proj_ok else "Project name not detected")

    # Copyright
    cr_ok = has_any(text_lc, [r"copyright"]) and bool(re.search(r"(19|20)\d{2}", text))
    add_row(rows, git_url, "notice_check", "Copyright – short one-liner",
            "PASS" if cr_ok else "WARN",
            "Copyright with year detected" if cr_ok else "No copyright with year")

    # Licence declaration
    lic_ok = has_any(text_lc, [r"licen[cs]e"]) and (has_any(text_lc, [r"https?://", r"\bLICENSE\b", r"\bCOPYING\b"]))
    add_row(rows, git_url, "notice_check", "Licence – declaration/link",
            "PASS" if lic_ok else "WARN",
            "Licence declaration detected" if lic_ok else "No licence declaration")

    # Authors/contributors
    auth_ok = has_any(text_lc, [r"\bauthors?\b", r"\bcontributors?\b", r"\bAUTHORS\b"])
    add_row(rows, git_url, "notice_check", "Authors and contributors – link",
            "PASS" if auth_ok else "WARN",
            "Authors reference detected" if auth_ok else "No authors reference")

    # Third-party components
    third_ok = has_any(text_lc, [r"third[-\s]?party", r"dependencies"]) or (text_lc.count("license") >= 2)
    add_row(rows, git_url, "notice_check", "Third-party components",
            "PASS" if third_ok else "WARN",
            "Third-party section detected" if third_ok else "No third-party components listing")

    # Tools used
    tools_ok = has_any(text_lc, [r"\btools?\b", r"\bbuilt with\b"])
    add_row(rows, git_url, "notice_check", "Tools used",
            "PASS" if tools_ok else "WARN",
            "Tools mention detected" if tools_ok else "No tools mention")

    # Trademark
    tm_ok = has_any(text_lc, [r"trademark"]) or ("™" in text) or ("®" in text)
    add_row(rows, git_url, "notice_check", "Trademark disclaimer",
            "PASS" if tm_ok else "WARN",
            "Trademark wording detected" if tm_ok else "No trademark wording")

    # Patents
    pat_ok = has_any(text_lc, [r"patent"])
    add_row(rows, git_url, "notice_check", "Patents statements",
            "PASS" if pat_ok else "WARN",
            "Patent wording detected" if pat_ok else "No patent wording")

    # Acknowledgements
    ack_ok = has_any(text_lc, [r"acknowledg", r"\bthanks\b", r"\bfunding\b"])
    add_row(rows, git_url, "notice_check", "Special acknowledgements",
            "PASS" if ack_ok else "WARN",
            "Acknowledgements detected" if ack_ok else "No acknowledgements")

    # Prior notices
    prior_ok = has_any(text_lc, [r"prior notice", r"this product includes", r"third[-\s]?party notices"])
    add_row(rows, git_url, "notice_check", "Prior notices for third-party components",
            "PASS" if prior_ok else "WARN",
            "Prior-notice wording detected" if prior_ok else "No prior-notice wording")

    return rows


# ============================================================================
# CHANGELOG CHECKS
# ============================================================================

def changelog_checks(repo_root: Path, git_url: str, repo_name: str) -> list[dict]:
    """CHANGELOG file content checks."""
    rows: list[dict] = []
    found = find_any(repo_root, ARTEFACTS["CHANGELOG"])

    if not found:
        add_row(rows, git_url, "changelog_check", "CHANGELOG content checks", "SKIP", "CHANGELOG not found")
        return rows

    cl_path = repo_root / found[0]
    text = read_text_safe(cl_path)
    text_lc = text.lower()

    # Project name
    proj_ok = repo_name.lower() in text_lc[:500]
    add_row(rows, git_url, "changelog_check", "Project name",
            "PASS" if proj_ok else "WARN",
            "Project name detected" if proj_ok else "Project name not detected")

    # Reverse chronological order (heuristic: latest version/date first)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    version_lines = [l for l in lines if re.search(r"v?\d+\.\d+(\.\d+)?|unreleased", l, re.IGNORECASE)]
    chrono_ok = len(version_lines) >= 1
    add_row(rows, git_url, "changelog_check", "Entries in reverse chronological order",
            "PASS" if chrono_ok else "WARN",
            "Version entries detected" if chrono_ok else "No clear version entries")

    # Version numbers and dates
    version_ok = bool(re.search(r"v?\d+\.\d+(\.\d+)?\s*-?\s*\d{4}", text))
    add_row(rows, git_url, "changelog_check", "Version numbers, dates and summaries",
            "PASS" if version_ok else "WARN",
            "Version + date pattern detected" if version_ok else "No clear version/date pattern")

    # Changelog sections
    sections = [
        ("Added – features or items", [r"^\s*##?\s*added\b", r"^\s*\*\*added\*\*"]),
        ("Changed – features or items", [r"^\s*##?\s*changed\b", r"^\s*\*\*changed\*\*"]),
        ("Deprecated – features or items", [r"^\s*##?\s*deprecated\b", r"^\s*\*\*deprecated\*\*"]),
        ("Removed – features or items", [r"^\s*##?\s*removed\b", r"^\s*\*\*removed\*\*"]),
        ("Fixed – bugs or issues", [r"^\s*##?\s*fixed\b", r"^\s*\*\*fixed\*\*"]),
        ("Security – updates or patches", [r"^\s*##?\s*security\b", r"^\s*\*\*security\*\*"]),
    ]

    for name, pats in sections:
        ok = any(re.search(p, text_lc, flags=re.MULTILINE | re.IGNORECASE) for p in pats)
        add_row(rows, git_url, "changelog_check", name,
                "PASS" if ok else "WARN",
                f"{name.split(' – ')[0]} section detected" if ok else f"No {name.split(' – ')[0]} section")

    return rows


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python check_repo_artefacts.py <git_url>")
        print("Example: python check_repo_artefacts.py https://github.com/GEANT/CAT.git")
        return 2

    git_url = sys.argv[1].strip()
    csv_file = Path("artefact_check.csv")
    rows = []

    with tempfile.TemporaryDirectory(prefix="artefact_check_") as tmpdir:
        repo_dir = Path(tmpdir) / "repo"

        print(f"Cloning: {git_url}")
        try:
            run(["git", "clone", "--depth", "1", git_url, str(repo_dir)])
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 2

        if not (repo_dir / ".git").exists():
            print("ERROR: Clone did not produce a git repository.")
            return 2

        repo_name = repo_name_from_url(git_url)
        print(f"Repository name: {repo_name}\n")

        # ===== KEY ARTEFACTS PRESENCE =====
        print("Checking key artifacts...")
        found_map = {}
        for artefact, candidates in ARTEFACTS.items():
            found = find_any(repo_dir, candidates)
            found_map[artefact] = found
            status = "PASS" if found else "FAIL"
            add_row(rows, git_url, "key_artefact", artefact, status, ", ".join(found) if found else "Not found")

        # ===== BINARY FILES =====
        print("Scanning for binary files...")
        bin_rows = scan_binaries(repo_dir)
        if not bin_rows:
            add_row(rows, git_url, "binary_file", "Binary artefacts", "PASS", "No binary files detected")
        else:
            for r in bin_rows:
                add_row(rows, git_url, r["check_type"], r["path"], "FAIL",
                        f'{r["reason"]}; size={r["size_bytes"]}')

        # ===== LARGE FILES =====
        print("Scanning for large files...")
        large_rows = scan_large_files(repo_dir, LARGE_FILE_THRESHOLD_BYTES)
        if not large_rows:
            add_row(rows, git_url, "large_file", f"Large files (>= {LARGE_FILE_THRESHOLD_BYTES} bytes)",
                    "PASS", "No large files detected")
        else:
            for r in large_rows:
                add_row(rows, git_url, r["check_type"], r["path"], "WARN",
                        f'{r["reason"]}; size={r["size_bytes"]}')

        # ===== CONTENT CHECKS =====
        print("Analyzing README content...")
        rows.extend(readme_checks(repo_dir, git_url, repo_name))

        print("Analyzing LICENSE content...")
        rows.extend(license_checks(repo_dir, git_url))

        print("Analyzing COPYRIGHT content...")
        rows.extend(copyright_checks(repo_dir, git_url))

        print("Analyzing AUTHORS content...")
        rows.extend(authors_checks(repo_dir, git_url))

        print("Analyzing NOTICE content...")
        rows.extend(notice_checks(repo_dir, git_url, repo_name))

        print("Analyzing CHANGELOG content...")
        rows.extend(changelog_checks(repo_dir, git_url, repo_name))

        # ===== CONDITIONAL CHECKS =====
        # Contributing fallback
        if not found_map.get("CONTRIBUTING"):
            rp = readme_path(repo_dir)
            if rp:
                md = read_text_safe(rp).lower()
                contrib_ok = bool(re.search(r"\bcontributing\b", md))
                add_row(rows, git_url, "conditional_check", "Contributing (fallback to README)",
                        "PASS" if contrib_ok else "FAIL",
                        "Found in README" if contrib_ok else "Not in CONTRIBUTING file or README")
            else:
                add_row(rows, git_url, "conditional_check", "Contributing (fallback to README)",
                        "FAIL", "No CONTRIBUTING file and README missing")

        # Authors fallback
        if not found_map.get("AUTHORS"):
            rp = readme_path(repo_dir)
            if rp:
                md = read_text_safe(rp).lower()
                authors_ok = bool(re.search(r"\bauthor|contributors?\b", md))
                add_row(rows, git_url, "conditional_check", "Authors (fallback to README)",
                        "PASS" if authors_ok else "WARN",
                        "Found in README" if authors_ok else "Not in AUTHORS file or README")

    # ===== WRITE RESULTS =====
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["repo_url", "check_type", "item", "status", "details"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 70}")
    print(f"CSV report written to: {csv_file.resolve()}")
    print(f"Total checks performed: {len(rows)}")

    # Summary statistics
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    warn_count = sum(1 for r in rows if r["status"] == "WARN")
    fail_count = sum(1 for r in rows if r["status"] == "FAIL")
    skip_count = sum(1 for r in rows if r["status"] == "SKIP")

    print(f"\nResults Summary:")
    print(f"  PASS: {pass_count}")
    print(f"  WARN: {warn_count}")
    print(f"  FAIL: {fail_count}")
    print(f"  SKIP: {skip_count}")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())