from __future__ import annotations
import sys
import subprocess
import tempfile
import csv
import json
import re
from pathlib import Path
from typing import Optional, Iterable
from dataclasses import dataclass, field
from enum import Enum

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
# PROBLEM LOGGING SYSTEM
# ============================================================================

class Severity(Enum):
    """Problem severity levels"""
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class ProblemCode(Enum):
    """Standardized problem codes for the Software Artefacts Checklist"""

    # Key artefacts
    ARTEFACT_README_MISSING = "ARTEFACT_README_MISSING"
    ARTEFACT_LICENSE_MISSING = "ARTEFACT_LICENSE_MISSING"
    ARTEFACT_COPYRIGHT_MISSING = "ARTEFACT_COPYRIGHT_MISSING"
    ARTEFACT_AUTHORS_MISSING = "ARTEFACT_AUTHORS_MISSING"
    ARTEFACT_NOTICE_MISSING = "ARTEFACT_NOTICE_MISSING"
    ARTEFACT_CHANGELOG_MISSING = "ARTEFACT_CHANGELOG_MISSING"
    ARTEFACT_CONTRIBUTING_MISSING = "ARTEFACT_CONTRIBUTING_MISSING"

    # Binary and large files
    BINARY_FILE_DETECTED = "BINARY_FILE_DETECTED"
    LARGE_FILE_DETECTED = "LARGE_FILE_DETECTED"

    # README checks
    README_PROJECT_NAME_MISSING = "README_PROJECT_NAME_MISSING"
    README_COPYRIGHT_MISSING = "README_COPYRIGHT_MISSING"
    README_STATUS_MISSING = "README_STATUS_MISSING"
    README_TAGS_MISSING = "README_TAGS_MISSING"
    README_VERSION_MISSING = "README_VERSION_MISSING"
    README_LICENSE_ONELINER_MISSING = "README_LICENSE_ONELINER_MISSING"
    README_BADGES_MISSING = "README_BADGES_MISSING"
    README_BADGE_METADATA_MISSING = "README_BADGE_METADATA_MISSING"
    README_BADGE_DOWNLOADS_MISSING = "README_BADGE_DOWNLOADS_MISSING"
    README_BADGE_COMMUNITY_MISSING = "README_BADGE_COMMUNITY_MISSING"
    README_BADGE_CI_MISSING = "README_BADGE_CI_MISSING"
    README_BADGE_QUALITY_MISSING = "README_BADGE_QUALITY_MISSING"
    README_BADGE_COVERAGE_MISSING = "README_BADGE_COVERAGE_MISSING"
    README_BADGE_SECURITY_MISSING = "README_BADGE_SECURITY_MISSING"
    README_BADGE_DOCS_MISSING = "README_BADGE_DOCS_MISSING"
    README_BADGE_SUPPORT_MISSING = "README_BADGE_SUPPORT_MISSING"
    README_DESCRIPTION_MISSING = "README_DESCRIPTION_MISSING"
    README_FEATURES_MISSING = "README_FEATURES_MISSING"
    README_SCOPE_MISSING = "README_SCOPE_MISSING"
    README_COMPATIBILITY_MISSING = "README_COMPATIBILITY_MISSING"
    README_STRUCTURE_MISSING = "README_STRUCTURE_MISSING"
    README_REQUIREMENTS_MISSING = "README_REQUIREMENTS_MISSING"
    README_INSTALLATION_MISSING = "README_INSTALLATION_MISSING"
    README_USAGE_MISSING = "README_USAGE_MISSING"
    README_DOCUMENTATION_MISSING = "README_DOCUMENTATION_MISSING"
    README_VERSION_CONTROL_MISSING = "README_VERSION_CONTROL_MISSING"
    README_TROUBLESHOOTING_MISSING = "README_TROUBLESHOOTING_MISSING"
    README_SUPPORT_INFO_MISSING = "README_SUPPORT_INFO_MISSING"
    README_PRIVACY_MISSING = "README_PRIVACY_MISSING"
    README_ROADMAP_MISSING = "README_ROADMAP_MISSING"
    README_AUTHORS_MISSING = "README_AUTHORS_MISSING"
    README_CONTRIBUTING_MISSING = "README_CONTRIBUTING_MISSING"
    README_FUNDING_MISSING = "README_FUNDING_MISSING"
    README_ACKNOWLEDGEMENTS_MISSING = "README_ACKNOWLEDGEMENTS_MISSING"
    README_DEPENDENCIES_MISSING = "README_DEPENDENCIES_MISSING"
    README_TOOLS_MISSING = "README_TOOLS_MISSING"
    README_LICENSE_PARAGRAPH_MISSING = "README_LICENSE_PARAGRAPH_MISSING"
    README_COPYRIGHT_DETAILS_MISSING = "README_COPYRIGHT_DETAILS_MISSING"
    README_BROKEN_LINKS = "README_BROKEN_LINKS"

    # LICENSE checks
    LICENSE_FILE_TOO_SHORT = "LICENSE_FILE_TOO_SHORT"
    LICENSE_TYPE_UNKNOWN = "LICENSE_TYPE_UNKNOWN"
    LICENSE_SCANCODE_NOT_AVAILABLE = "LICENSE_SCANCODE_NOT_AVAILABLE"
    LICENSE_SCANCODE_FAILED = "LICENSE_SCANCODE_FAILED"
    LICENSE_MISMATCH_DETECTED = "LICENSE_MISMATCH_DETECTED"
    LICENSE_UNDECLARED_FOUND = "LICENSE_UNDECLARED_FOUND"
    LICENSE_DECLARED_NOT_FOUND = "LICENSE_DECLARED_NOT_FOUND"

    # COPYRIGHT checks
    COPYRIGHT_STATEMENT_MISSING = "COPYRIGHT_STATEMENT_MISSING"
    COPYRIGHT_GEANT_MISSING = "COPYRIGHT_GEANT_MISSING"
    COPYRIGHT_STATEMENTS_COUNT = "COPYRIGHT_STATEMENTS_COUNT"
    COPYRIGHT_EU_LOGO_MISSING = "COPYRIGHT_EU_LOGO_MISSING"

    # AUTHORS checks
    AUTHORS_GEANT_PHASE_MISSING = "AUTHORS_GEANT_PHASE_MISSING"
    AUTHORS_CONTACTS_MISSING = "AUTHORS_CONTACTS_MISSING"
    AUTHORS_CONTRIBUTORS_MISSING = "AUTHORS_CONTRIBUTORS_MISSING"
    AUTHORS_FUNDING_MISSING = "AUTHORS_FUNDING_MISSING"

    # NOTICE checks
    NOTICE_PROJECT_NAME_MISSING = "NOTICE_PROJECT_NAME_MISSING"
    NOTICE_COPYRIGHT_MISSING = "NOTICE_COPYRIGHT_MISSING"
    NOTICE_LICENSE_MISSING = "NOTICE_LICENSE_MISSING"
    NOTICE_AUTHORS_MISSING = "NOTICE_AUTHORS_MISSING"
    NOTICE_THIRD_PARTY_MISSING = "NOTICE_THIRD_PARTY_MISSING"
    NOTICE_TOOLS_MISSING = "NOTICE_TOOLS_MISSING"
    NOTICE_TRADEMARK_MISSING = "NOTICE_TRADEMARK_MISSING"
    NOTICE_PATENTS_MISSING = "NOTICE_PATENTS_MISSING"
    NOTICE_ACKNOWLEDGEMENTS_MISSING = "NOTICE_ACKNOWLEDGEMENTS_MISSING"
    NOTICE_PRIOR_NOTICES_MISSING = "NOTICE_PRIOR_NOTICES_MISSING"

    # CHANGELOG checks
    CHANGELOG_PROJECT_NAME_MISSING = "CHANGELOG_PROJECT_NAME_MISSING"
    CHANGELOG_NOT_CHRONOLOGICAL = "CHANGELOG_NOT_CHRONOLOGICAL"
    CHANGELOG_VERSION_DATE_MISSING = "CHANGELOG_VERSION_DATE_MISSING"
    CHANGELOG_ADDED_MISSING = "CHANGELOG_ADDED_MISSING"
    CHANGELOG_CHANGED_MISSING = "CHANGELOG_CHANGED_MISSING"
    CHANGELOG_DEPRECATED_MISSING = "CHANGELOG_DEPRECATED_MISSING"
    CHANGELOG_REMOVED_MISSING = "CHANGELOG_REMOVED_MISSING"
    CHANGELOG_FIXED_MISSING = "CHANGELOG_FIXED_MISSING"
    CHANGELOG_SECURITY_MISSING = "CHANGELOG_SECURITY_MISSING"

    # Conditional checks
    CONTRIBUTING_FALLBACK_MISSING = "CONTRIBUTING_FALLBACK_MISSING"
    AUTHORS_FALLBACK_MISSING = "AUTHORS_FALLBACK_MISSING"


# Default priorities for problem codes (1=highest, 5=lowest)
DEFAULT_PRIORITIES = {
    ProblemCode.ARTEFACT_LICENSE_MISSING: 1,
    ProblemCode.ARTEFACT_README_MISSING: 1,
    ProblemCode.ARTEFACT_COPYRIGHT_MISSING: 2,
    ProblemCode.BINARY_FILE_DETECTED: 1,
    ProblemCode.COPYRIGHT_GEANT_MISSING: 3,
    ProblemCode.COPYRIGHT_EU_LOGO_MISSING: 3,
    ProblemCode.LICENSE_FILE_TOO_SHORT: 2,
    ProblemCode.LICENSE_MISMATCH_DETECTED: 1,
    ProblemCode.LICENSE_UNDECLARED_FOUND: 2,
    ProblemCode.README_BROKEN_LINKS: 2,
    ProblemCode.AUTHORS_FUNDING_MISSING: 3,
}


@dataclass
class Problem:
    """Represents a detected problem or finding"""
    code: ProblemCode
    severity: Severity
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    priority: Optional[int] = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        # Set default priority if not provided
        if self.priority is None:
            self.priority = DEFAULT_PRIORITIES.get(self.code, 3)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "priority": self.priority,
            "details": self.details
        }


class ProblemLogger:
    """Centralized problem logging system"""

    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.problems: list[Problem] = []

    def log_problem(
            self,
            code: ProblemCode,
            message: str,
            severity: Severity = Severity.WARN,
            file_path: Optional[str] = None,
            line_number: Optional[int] = None,
            priority: Optional[int] = None,
            **details
    ) -> None:
        """Log a problem with optional details"""
        problem = Problem(
            code=code,
            severity=severity,
            message=message,
            file_path=file_path,
            line_number=line_number,
            priority=priority,
            details=details
        )
        self.problems.append(problem)

    def log_pass(
            self,
            code: ProblemCode,
            message: str,
            **details
    ) -> None:
        """Log a successful check"""
        self.log_problem(code, message, Severity.PASS, **details)

    def get_problems(self, severity: Optional[Severity] = None) -> list[Problem]:
        """Get all problems, optionally filtered by severity"""
        if severity:
            return [p for p in self.problems if p.severity == severity]
        return self.problems

    def to_csv(self, filepath: Path) -> None:
        """Export problems to CSV format"""
        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["repo_url", "code", "severity", "message", "file_path", "line_number", "priority",
                            "details"]
            )
            writer.writeheader()
            for problem in self.problems:
                row = {
                    "repo_url": self.repo_url,
                    "code": problem.code.value,
                    "severity": problem.severity.value,
                    "message": problem.message,
                    "file_path": problem.file_path or "",
                    "line_number": problem.line_number or "",
                    "priority": problem.priority,
                    "details": json.dumps(problem.details) if problem.details else ""
                }
                writer.writerow(row)

    def to_json(self, filepath: Path) -> None:
        """Export problems to JSON format"""
        data = {
            "repo_url": self.repo_url,
            "problems": [p.to_dict() for p in self.problems],
            "summary": self.get_summary()
        }
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> dict:
        """Get summary statistics"""
        return {
            "total": len(self.problems),
            "pass": len(self.get_problems(Severity.PASS)),
            "info": len(self.get_problems(Severity.INFO)),
            "warn": len(self.get_problems(Severity.WARN)),
            "fail": len(self.get_problems(Severity.FAIL)),
            "skip": len(self.get_problems(Severity.SKIP)),
        }


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


def check_scancode_available() -> bool:
    """Check if scancode-toolkit is available."""
    try:
        result = subprocess.run(
            ["scancode", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_scancode_license_scan(repo_root: Path) -> Optional[dict]:
    """
    Run scancode to detect licenses in the repository.
    Returns a dict with license information or None if scan fails.
    """
    try:
        import tempfile

        # Create a temporary file for JSON output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            output_file = Path(tmp.name)

        try:
            # Run scancode with license detection
            # --license: detect licenses
            # --license-text: include license text
            # --json-pp: output as pretty-printed JSON (easier to debug)
            # --quiet: suppress progress (removed to see what's happening)
            # --timeout: set timeout per file
            cmd = [
                "scancode",
                "--license",
                "--license-text",
                "--json-pp", str(output_file),
                "--timeout", "30",
                "--max-depth", "10",
                str(repo_root)
            ]

            print(f"Running ScanCode license detection...")
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"ScanCode failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None

            # Read and parse the JSON output
            if not output_file.exists():
                print(f"Error: Output file not created at {output_file}")
                return None

            print(f"Reading ScanCode output from {output_file}")
            with output_file.open('r', encoding='utf-8') as f:
                scan_data = json.load(f)

            # Debug: print some info about what we got
            if scan_data:
                print(f"ScanCode output keys: {list(scan_data.keys())}")
                if 'files' in scan_data:
                    print(f"Number of files in results: {len(scan_data['files'])}")
                    # Show first file with license for debugging
                    for file_info in scan_data['files'][:10]:
                        if file_info.get('licenses'):
                            print(f"Sample: {file_info.get('path')} -> {file_info.get('licenses')}")
                            break

            return scan_data

        finally:
            # Clean up temp file
            if output_file.exists():
                try:
                    output_file.unlink()
                except:
                    pass  # Ignore cleanup errors

    except subprocess.TimeoutExpired:
        print("ScanCode scan timed out after 10 minutes")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ScanCode JSON output: {e}")
        return None
    except Exception as e:
        print(f"Error running ScanCode: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_licenses_from_scancode(scan_data: dict) -> dict:
    """
    Extract license information from ScanCode output.
    Returns dict with:
    - detected_licenses: set of SPDX license identifiers found
    - license_files: dict mapping file paths to licenses found in them
    - main_license: most common license (likely the project's main license)
    """
    license_count = {}
    license_files = {}

    if not scan_data:
        return {
            'detected_licenses': set(),
            'license_files': {},
            'main_license': None,
            'license_count': {}
        }

    # ScanCode output structure can vary, check both 'files' and 'headers'
    files_data = scan_data.get('files', [])

    if not files_data:
        print(f"Warning: No 'files' key in ScanCode output. Keys found: {list(scan_data.keys())}")
        return {
            'detected_licenses': set(),
            'license_files': {},
            'main_license': None,
            'license_count': {}
        }

    print(f"Processing {len(files_data)} files from ScanCode results...")

    for file_info in files_data:
        # Skip directories
        if file_info.get('type') == 'directory':
            continue

        licenses = file_info.get('licenses', [])
        license_detections = file_info.get('license_detections', [])

        # Try both 'licenses' and 'license_detections' fields
        all_licenses = licenses if licenses else license_detections

        if not all_licenses:
            continue

        file_path = file_info.get('path', '')

        for lic in all_licenses:
            # Try multiple ways to get the license identifier
            spdx_key = (
                    lic.get('spdx_license_key') or
                    lic.get('key') or
                    lic.get('license_expression') or
                    lic.get('matched_rule', {}).get('license_expression') or
                    lic.get('short_name')
            )

            if spdx_key:
                # Normalize license name
                spdx_key = normalize_license_name(spdx_key)
                license_count[spdx_key] = license_count.get(spdx_key, 0) + 1

                if file_path not in license_files:
                    license_files[file_path] = []
                if spdx_key not in license_files[file_path]:
                    license_files[file_path].append(spdx_key)

    # Determine main license (most frequently occurring)
    main_license = None
    if license_count:
        main_license = max(license_count.items(), key=lambda x: x[1])[0]
        print(f"Found {len(license_count)} unique licenses, main license: {main_license}")
    else:
        print("Warning: No licenses extracted from ScanCode results")

    return {
        'detected_licenses': set(license_count.keys()),
        'license_files': license_files,
        'main_license': main_license,
        'license_count': license_count
    }


def normalize_license_name(name: str) -> str:
    """Normalize license name for comparison."""
    if not name:
        return ""

    name = name.upper().strip()

    # Map common variations to standard names
    mappings = {
        'MIT': 'MIT',
        'MIT LICENSE': 'MIT',
        'APACHE': 'APACHE-2.0',
        'APACHE-2.0': 'APACHE-2.0',
        'APACHE 2.0': 'APACHE-2.0',
        'APACHE LICENSE 2.0': 'APACHE-2.0',
        'APACHE-2.0-ONLY': 'APACHE-2.0',
        'GPL': 'GPL-3.0',
        'GPL-3.0': 'GPL-3.0',
        'GPL-3.0-ONLY': 'GPL-3.0',
        'GPL-3.0-OR-LATER': 'GPL-3.0+',
        'GPL-2.0': 'GPL-2.0',
        'GPL-2.0-ONLY': 'GPL-2.0',
        'GPL-2.0-OR-LATER': 'GPL-2.0+',
        'GPLV3': 'GPL-3.0',
        'GPLV2': 'GPL-2.0',
        'BSD': 'BSD-3-CLAUSE',
        'BSD-3-CLAUSE': 'BSD-3-CLAUSE',
        'BSD-2-CLAUSE': 'BSD-2-CLAUSE',
        'EUPL': 'EUPL-1.2',
        'EUPL-1.2': 'EUPL-1.2',
        'EUPL-1.1': 'EUPL-1.1',
        'LGPL': 'LGPL-3.0',
        'LGPL-3.0': 'LGPL-3.0',
        'LGPL-2.1': 'LGPL-2.1',
        'LGPL-3.0-ONLY': 'LGPL-3.0',
        'LGPL-2.1-ONLY': 'LGPL-2.1',
    }

    # Direct match
    if name in mappings:
        return mappings[name]

    # Partial match
    for key, value in mappings.items():
        if key in name or name in key:
            return value

    # Return as-is if no mapping found
    return name


# ============================================================================
# BINARY & LARGE FILE CHECKS
# ============================================================================

def scan_binaries(repo_root: Path, logger: ProblemLogger) -> None:
    """Find binary/compiled files."""
    found_any = False
    for f in iter_files(repo_root):
        ext = f.suffix.lower()
        if ext in BINARY_EXTENSIONS:
            found_any = True
            size = f.stat().st_size
            rel_path = str(f.relative_to(repo_root)).replace("\\", "/")
            logger.log_problem(
                ProblemCode.BINARY_FILE_DETECTED,
                f"Binary file detected: {rel_path}",
                Severity.FAIL,
                file_path=rel_path,
                extension=ext,
                size_bytes=size
            )

    if not found_any:
        logger.log_pass(
            ProblemCode.BINARY_FILE_DETECTED,
            "No binary files detected"
        )


def scan_large_files(repo_root: Path, threshold: int, logger: ProblemLogger) -> None:
    """Find files exceeding size threshold."""
    found_any = False
    for f in iter_files(repo_root):
        size = f.stat().st_size
        if size >= threshold:
            found_any = True
            rel_path = str(f.relative_to(repo_root)).replace("\\", "/")
            logger.log_problem(
                ProblemCode.LARGE_FILE_DETECTED,
                f"Large file detected: {rel_path}",
                Severity.WARN,
                file_path=rel_path,
                size_bytes=size,
                threshold_bytes=threshold
            )

    if not found_any:
        logger.log_pass(
            ProblemCode.LARGE_FILE_DETECTED,
            f"No large files detected (>= {threshold} bytes)"
        )


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


def readme_checks(repo_root: Path, repo_name: str, logger: ProblemLogger) -> None:
    """Comprehensive README content checks."""
    rp = readme_path(repo_root)

    if rp is None:
        logger.log_problem(
            ProblemCode.ARTEFACT_README_MISSING,
            "README file not found",
            Severity.SKIP
        )
        return

    md = read_text_safe(rp)
    text_lc = md.lower()
    headings = [h.lower() for h in extract_headings(md)]
    repo_name_lc = repo_name.lower()

    # Project name
    title_ok = (repo_name_lc in text_lc[:4000]) or any(repo_name_lc in h for h in headings)
    if title_ok:
        logger.log_pass(ProblemCode.README_PROJECT_NAME_MISSING, "Project name detected in README")
    else:
        logger.log_problem(
            ProblemCode.README_PROJECT_NAME_MISSING,
            "Project name not clearly detected in README",
            Severity.WARN,
            file_path="README"
        )

    # Copyright one-liner
    cr_ok = bool(re.search(r"copyright\s*(?:\(|©)?\s*(19|20)\d{2}", text_lc))
    if cr_ok:
        logger.log_pass(ProblemCode.README_COPYRIGHT_MISSING, "Copyright statement detected in README")
    else:
        logger.log_problem(
            ProblemCode.README_COPYRIGHT_MISSING,
            "No clear copyright one-liner detected in README",
            Severity.WARN,
            file_path="README"
        )

    # Project status
    status_ok = has_any(text_lc, [r"\bstatus\b", r"\blifecycle\b", r"\bbeta\b", r"\bstable\b",
                                  r"\bretired\b", r"\bdeprecated\b", r"\bdevelopment\b"])
    if status_ok:
        logger.log_pass(ProblemCode.README_STATUS_MISSING, "Project status/lifecycle detected")
    else:
        logger.log_problem(
            ProblemCode.README_STATUS_MISSING,
            "No explicit lifecycle/status detected in README",
            Severity.WARN,
            file_path="README"
        )

    # Tags
    tags_ok = has_any(text_lc, [r"^\s*tags?\s*[:\-]", r"\bkeywords?\b"])
    if tags_ok:
        logger.log_pass(ProblemCode.README_TAGS_MISSING, "Tags/keywords detected")
    else:
        logger.log_problem(
            ProblemCode.README_TAGS_MISSING,
            "No tags/keywords detected in README",
            Severity.WARN,
            file_path="README"
        )

    # Version
    version_ok = has_any(text_lc, [r"\bversion\s*v?\d+\.\d+(\.\d+)?", r"\brelease\s*v?\d+\.\d+"])
    if version_ok:
        logger.log_pass(ProblemCode.README_VERSION_MISSING, "Version indication detected")
    else:
        logger.log_problem(
            ProblemCode.README_VERSION_MISSING,
            "No clear version indication in README",
            Severity.WARN,
            file_path="README"
        )

    # Licence one-liner
    lic_ok = has_any(text_lc, [r"\blicen[cs]ed\s+under\b", r"\blicen[cs]e\s*[:\-]",
                               r"\bMIT\b|\bApache\b|\bGPL\b|\bEUPL\b|\bBSD\b"]) and \
             has_any(text_lc, [r"\bLICENSE\b", r"\bCOPYING\b", r"https?://"])
    if lic_ok:
        logger.log_pass(ProblemCode.README_LICENSE_ONELINER_MISSING, "Licence one-liner with link detected")
    else:
        logger.log_problem(
            ProblemCode.README_LICENSE_ONELINER_MISSING,
            "Licence one-liner/link not detected in README",
            Severity.WARN,
            file_path="README"
        )

    # Badges
    badge_count = len(MD_BADGE_RE.findall(md))
    if badge_count > 0:
        logger.log_pass(ProblemCode.README_BADGES_MISSING, f"Found {badge_count} badge(s)", badge_count=badge_count)
    else:
        logger.log_problem(
            ProblemCode.README_BADGES_MISSING,
            "No badges detected in README",
            Severity.WARN,
            file_path="README"
        )

    # Badge categories
    badge_checks = [
        (ProblemCode.README_BADGE_METADATA_MISSING, "Project metadata and release status",
         [r"release", r"tag", r"version", r"latest"]),
        (ProblemCode.README_BADGE_DOWNLOADS_MISSING, "Package downloads and versions",
         [r"pypi", r"npm", r"maven", r"nuget", r"downloads"]),
        (ProblemCode.README_BADGE_COMMUNITY_MISSING, "Community engagement",
         [r"contributors", r"contributing", r"prs\s+welcome"]),
        (ProblemCode.README_BADGE_CI_MISSING, "CI/CD workflows",
         [r"github\.com/.*/actions", r"travis", r"circleci", r"build\s+status"]),
        (ProblemCode.README_BADGE_QUALITY_MISSING, "Code quality",
         [r"codeql", r"sonar", r"codeclimate", r"quality"]),
        (ProblemCode.README_BADGE_COVERAGE_MISSING, "Test coverage",
         [r"codecov", r"coveralls", r"coverage"]),
        (ProblemCode.README_BADGE_SECURITY_MISSING, "Security vulnerabilities",
         [r"snyk", r"dependabot", r"vulnerab", r"security"]),
        (ProblemCode.README_BADGE_DOCS_MISSING, "Documentation",
         [r"readthedocs", r"docs", r"documentation\s+status"]),
        (ProblemCode.README_BADGE_SUPPORT_MISSING, "Community support",
         [r"slack", r"discord", r"discourse", r"gitter"]),
    ]

    for code, name, pats in badge_checks:
        ok = any(re.search(p, md, flags=re.IGNORECASE) for p in pats)
        if ok:
            logger.log_pass(code, f"Badge detected: {name}")
        else:
            logger.log_problem(code, f"Badge not detected: {name}", Severity.WARN, file_path="README")

    # Description
    desc_ok = len(text_lc) >= 100 and not text_lc.strip().startswith("#")
    if desc_ok:
        logger.log_pass(ProblemCode.README_DESCRIPTION_MISSING, "Description/overview detected")
    else:
        logger.log_problem(
            ProblemCode.README_DESCRIPTION_MISSING,
            "No clear description/overview in README",
            Severity.WARN,
            file_path="README"
        )

    # Features
    feat_ok = ("features" in headings) or has_any(text_lc, [r"^\s*features?\s*[:\-]", r"## features"])
    if feat_ok:
        logger.log_pass(ProblemCode.README_FEATURES_MISSING, "Features section detected")
    else:
        logger.log_problem(
            ProblemCode.README_FEATURES_MISSING,
            "No features section in README",
            Severity.WARN,
            file_path="README"
        )

    # Scope
    scope_ok = any(h in headings for h in ["scope", "use cases", "limitations", "constraints"]) or \
               has_any(text_lc, [r"\bscope\b", r"\buse\s+case", r"\blimitations?\b"])
    if scope_ok:
        logger.log_pass(ProblemCode.README_SCOPE_MISSING, "Scope/use-case signals detected")
    else:
        logger.log_problem(
            ProblemCode.README_SCOPE_MISSING,
            "No clear scope section in README",
            Severity.WARN,
            file_path="README"
        )

    # Supported environments
    supp_ok = any(h in headings for h in ["supported", "compatibility", "platforms"]) or \
              has_any(text_lc, [r"\bsupported\b", r"\bcompatib", r"\bplatforms?\b"])
    if supp_ok:
        logger.log_pass(ProblemCode.README_COMPATIBILITY_MISSING, "Compatibility signals detected")
    else:
        logger.log_problem(
            ProblemCode.README_COMPATIBILITY_MISSING,
            "No compatibility section in README",
            Severity.WARN,
            file_path="README"
        )

    # Components/structure
    comp_ok = any(h in headings for h in ["architecture", "components", "structure"]) or \
              has_any(text_lc, [r"\barchitecture\b", r"\bcomponents?\b", r"\bproject\s+structure\b"])
    if comp_ok:
        logger.log_pass(ProblemCode.README_STRUCTURE_MISSING, "Structure signals detected")
    else:
        logger.log_problem(
            ProblemCode.README_STRUCTURE_MISSING,
            "No structure section in README",
            Severity.WARN,
            file_path="README"
        )

    # System requirements
    req_ok = any(h in headings for h in ["requirements", "prerequisites"]) or \
             has_any(text_lc, [r"\brequirements?\b", r"\bprerequisites?\b"])
    if req_ok:
        logger.log_pass(ProblemCode.README_REQUIREMENTS_MISSING, "Requirements detected")
    else:
        logger.log_problem(
            ProblemCode.README_REQUIREMENTS_MISSING,
            "No requirements section in README",
            Severity.WARN,
            file_path="README"
        )

    # Installation
    inst_ok = any(h in headings for h in ["installation", "install", "setup", "getting started"]) or \
              has_any(text_lc, [r"\binstall", r"\bsetup\b", r"\bgetting started\b"])
    if inst_ok:
        logger.log_pass(ProblemCode.README_INSTALLATION_MISSING, "Installation signals detected")
    else:
        logger.log_problem(
            ProblemCode.README_INSTALLATION_MISSING,
            "No installation section in README",
            Severity.WARN,
            file_path="README"
        )

    # Usage
    usage_ok = any(h in headings for h in ["usage", "how to use", "examples", "quickstart"]) or \
               has_any(text_lc, [r"\busage\b", r"\bexample", r"\bquickstart\b"])
    codeblock_ok = "```" in md
    if usage_ok and codeblock_ok:
        logger.log_pass(ProblemCode.README_USAGE_MISSING, "Usage + code blocks detected")
    else:
        logger.log_problem(
            ProblemCode.README_USAGE_MISSING,
            "Usage/examples not clearly present in README",
            Severity.WARN,
            file_path="README"
        )

    # Documentation
    docs_ok = any(h in headings for h in ["documentation", "docs"]) or \
              has_any(text_lc, [r"\bdocumentation\b", r"\bdocs/\b", r"readthedocs"])
    if docs_ok:
        logger.log_pass(ProblemCode.README_DOCUMENTATION_MISSING, "Documentation signals detected")
    else:
        logger.log_problem(
            ProblemCode.README_DOCUMENTATION_MISSING,
            "No documentation location in README",
            Severity.WARN,
            file_path="README"
        )

    # Version control
    vc_ok = has_any(text_lc, [r"\bbranch", r"\btag(s|ging)?", r"\brelease"])
    if vc_ok:
        logger.log_pass(ProblemCode.README_VERSION_CONTROL_MISSING, "Branch/tag wording detected")
    else:
        logger.log_problem(
            ProblemCode.README_VERSION_CONTROL_MISSING,
            "No branch/tagging conventions in README",
            Severity.WARN,
            file_path="README"
        )

    # Troubleshooting/FAQ
    faq_ok = any(h in headings for h in ["troubleshooting", "faq"]) or \
             has_any(text_lc, [r"\btroubleshoot", r"\bfaq\b"])
    if faq_ok:
        logger.log_pass(ProblemCode.README_TROUBLESHOOTING_MISSING, "Troubleshooting/FAQ detected")
    else:
        logger.log_problem(
            ProblemCode.README_TROUBLESHOOTING_MISSING,
            "No troubleshooting/FAQ in README",
            Severity.WARN,
            file_path="README"
        )

    # Support
    support_ok = has_any(text_lc, [r"\bissues\b", r"\bsupport\b", r"\bcontact\b"])
    if support_ok:
        logger.log_pass(ProblemCode.README_SUPPORT_INFO_MISSING, "Support/contact wording detected")
    else:
        logger.log_problem(
            ProblemCode.README_SUPPORT_INFO_MISSING,
            "No support info in README",
            Severity.WARN,
            file_path="README"
        )

    # Privacy policy
    privacy_ok = has_any(text_lc, [r"\bprivacy\b", r"privacy policy"])
    if privacy_ok:
        logger.log_pass(ProblemCode.README_PRIVACY_MISSING, "Privacy wording detected")
    else:
        logger.log_problem(
            ProblemCode.README_PRIVACY_MISSING,
            "No privacy policy in README",
            Severity.WARN,
            file_path="README"
        )

    # Roadmap
    roadmap_ok = any(h in headings for h in ["roadmap"]) or \
                 has_any(text_lc, [r"\broadmap\b", r"\bplanned\b", r"\bfuture\b"])
    if roadmap_ok:
        logger.log_pass(ProblemCode.README_ROADMAP_MISSING, "Roadmap detected")
    else:
        logger.log_problem(
            ProblemCode.README_ROADMAP_MISSING,
            "No roadmap in README",
            Severity.WARN,
            file_path="README"
        )

    # Authors mention
    authors_ok = has_any(text_lc, [r"\bauthors?\b", r"\bcontributors?\b"])
    if authors_ok:
        logger.log_pass(ProblemCode.README_AUTHORS_MISSING, "Authors mentioned")
    else:
        logger.log_problem(
            ProblemCode.README_AUTHORS_MISSING,
            "No authors mention in README",
            Severity.WARN,
            file_path="README"
        )

    # Contributing
    contrib_ok = any(h in headings for h in ["contributing"]) or \
                 has_any(text_lc, [r"\bcontributing\b", r"\bhow to contribute\b"])
    if contrib_ok:
        logger.log_pass(ProblemCode.README_CONTRIBUTING_MISSING, "Contributing detected")
    else:
        logger.log_problem(
            ProblemCode.README_CONTRIBUTING_MISSING,
            "No contributing guidance in README",
            Severity.WARN,
            file_path="README"
        )

    # Funding
    fund_ok = has_any(text_lc, [r"\bfunding\b", r"\bgrant\b", r"\bhorizon\b", r"\beu\b"])
    if fund_ok:
        logger.log_pass(ProblemCode.README_FUNDING_MISSING, "Funding wording detected")
    else:
        logger.log_problem(
            ProblemCode.README_FUNDING_MISSING,
            "No funding info in README",
            Severity.WARN,
            file_path="README"
        )

    # Acknowledgements
    ack_ok = has_any(text_lc, [r"\backnowledg", r"\bthanks\b"])
    if ack_ok:
        logger.log_pass(ProblemCode.README_ACKNOWLEDGEMENTS_MISSING, "Acknowledgements detected")
    else:
        logger.log_problem(
            ProblemCode.README_ACKNOWLEDGEMENTS_MISSING,
            "No acknowledgements in README",
            Severity.WARN,
            file_path="README"
        )

    # Dependencies
    dep_ok = has_any(text_lc, [r"\bdependencies\b", r"\brequirements\b"]) or \
             any((repo_root / f).exists() for f in
                 ["package.json", "requirements.txt", "pyproject.toml", "pom.xml", "go.mod"])
    if dep_ok:
        logger.log_pass(ProblemCode.README_DEPENDENCIES_MISSING, "Dependencies/manifests detected")
    else:
        logger.log_problem(
            ProblemCode.README_DEPENDENCIES_MISSING,
            "No dependencies section in README",
            Severity.WARN,
            file_path="README"
        )

    # Tools used
    tools_ok = has_any(text_lc, [r"\btools?\b", r"\bbuilt with\b"]) or \
               any((repo_root / f).exists() for f in ["Makefile", "Dockerfile"])
    if tools_ok:
        logger.log_pass(ProblemCode.README_TOOLS_MISSING, "Tools indicators detected")
    else:
        logger.log_problem(
            ProblemCode.README_TOOLS_MISSING,
            "No tools indicators in README",
            Severity.WARN,
            file_path="README"
        )

    # Licence paragraph
    lic_section = "license" in headings or has_any(text_lc, [r"^#+\s*licen[cs]e\b"])
    lic_para_ok = lic_section and has_any(text_lc, [r"\bLICENSE\b", r"https?://"])
    if lic_para_ok:
        logger.log_pass(ProblemCode.README_LICENSE_PARAGRAPH_MISSING, "Licence section detected")
    else:
        logger.log_problem(
            ProblemCode.README_LICENSE_PARAGRAPH_MISSING,
            "No licence summary in README",
            Severity.WARN,
            file_path="README"
        )

    # Copyright details
    cd_ok = has_any(text_lc, [r"\bCOPYRIGHT\b", r"copyright\s*(19|20)\d{2}"])
    if cd_ok:
        logger.log_pass(ProblemCode.README_COPYRIGHT_DETAILS_MISSING, "COPYRIGHT reference detected")
    else:
        logger.log_problem(
            ProblemCode.README_COPYRIGHT_DETAILS_MISSING,
            "No COPYRIGHT reference in README",
            Severity.WARN,
            file_path="README"
        )

    # Validate local links
    links = local_links_in_readme(md)
    broken = []
    for t in links:
        candidate = (repo_root / t).resolve()
        if str(candidate).startswith(str(repo_root.resolve())) and not candidate.exists():
            broken.append(t)

    if not broken:
        logger.log_pass(ProblemCode.README_BROKEN_LINKS, "All README local links are valid")
    else:
        logger.log_problem(
            ProblemCode.README_BROKEN_LINKS,
            f"Broken links detected in README: {', '.join(broken[:5])}",
            Severity.WARN,
            file_path="README",
            broken_links=broken
        )


# ============================================================================
# LICENSE CHECKS
# ============================================================================

def license_checks(repo_root: Path, logger: ProblemLogger, scancode_data: Optional[dict] = None) -> None:
    """LICENSE file content checks with ScanCode integration."""
    found = find_any(repo_root, ARTEFACTS["LICENSE"])

    if not found:
        logger.log_problem(
            ProblemCode.ARTEFACT_LICENSE_MISSING,
            "LICENSE file not found",
            Severity.SKIP
        )
        return

    lic_path = repo_root / found[0]
    text = read_text_safe(lic_path)

    # Full official license text
    full_text_ok = len(text) >= 500
    if full_text_ok:
        logger.log_pass(
            ProblemCode.LICENSE_FILE_TOO_SHORT,
            f"LICENSE file has adequate length ({len(text)} characters)",
            char_count=len(text)
        )
    else:
        logger.log_problem(
            ProblemCode.LICENSE_FILE_TOO_SHORT,
            "LICENSE file seems too short for full license text",
            Severity.WARN,
            file_path=found[0],
            char_count=len(text)
        )

    # License type detection - manual patterns (fallback)
    licenses_detected_manual = []
    license_patterns = {
        "MIT": r"\bMIT License\b",
        "Apache-2.0": r"\bApache License",
        "GPL-3.0": r"\bGNU General Public License",
        "BSD-3-Clause": r"\bBSD License",
        "EUPL-1.2": r"\bEuropean Union Public Licence",
    }

    for lic_name, pattern in license_patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            licenses_detected_manual.append(normalize_license_name(lic_name))

    # If ScanCode data is available, use it for more accurate detection
    if scancode_data:
        license_info = extract_licenses_from_scancode(scancode_data)
        detected_licenses = license_info['detected_licenses']
        main_license = license_info['main_license']
        license_count = license_info.get('license_count', {})

        # Check if LICENSE file declares the main license found by ScanCode
        declared_licenses = set(normalize_license_name(lic) for lic in licenses_detected_manual)

        if detected_licenses:
            logger.log_pass(
                ProblemCode.LICENSE_TYPE_UNKNOWN,
                f"ScanCode detected licenses: {', '.join(sorted(detected_licenses))}",
                licenses=list(detected_licenses),
                main_license=main_license,
                license_distribution=license_count
            )

            # Check for mismatches between declared and detected licenses
            if main_license:
                if main_license not in declared_licenses and declared_licenses:
                    logger.log_problem(
                        ProblemCode.LICENSE_MISMATCH_DETECTED,
                        f"LICENSE file declares {', '.join(declared_licenses)} but ScanCode detected {main_license} as main license",
                        Severity.WARN,
                        file_path=found[0],
                        declared=list(declared_licenses),
                        detected_main=main_license,
                        all_detected=list(detected_licenses)
                    )
                elif main_license in declared_licenses:
                    logger.log_pass(
                        ProblemCode.LICENSE_MISMATCH_DETECTED,
                        f"LICENSE file correctly declares {main_license}",
                        declared=list(declared_licenses),
                        detected=main_license
                    )

            # Check for licenses found in code but not declared
            undeclared = detected_licenses - declared_licenses
            if undeclared:
                # Filter out very common permissive licenses that might be in dependencies
                significant_undeclared = {
                    lic for lic in undeclared
                    if not any(x in lic for x in ['PUBLIC-DOMAIN', 'CC0', 'UNLICENSE'])
                }

                if significant_undeclared:
                    logger.log_problem(
                        ProblemCode.LICENSE_UNDECLARED_FOUND,
                        f"Licenses found in code but not declared in LICENSE: {', '.join(sorted(significant_undeclared))}",
                        Severity.WARN,
                        file_path=found[0],
                        undeclared_licenses=list(significant_undeclared),
                        hint="These may be from third-party dependencies"
                    )

            # Check for licenses declared but not found in scan
            if declared_licenses:
                not_found = declared_licenses - detected_licenses
                if not_found:
                    logger.log_problem(
                        ProblemCode.LICENSE_DECLARED_NOT_FOUND,
                        f"LICENSE declares {', '.join(not_found)} but ScanCode did not detect it in the codebase",
                        Severity.INFO,
                        file_path=found[0],
                        declared_but_not_found=list(not_found)
                    )
        else:
            logger.log_problem(
                ProblemCode.LICENSE_TYPE_UNKNOWN,
                "ScanCode did not detect any licenses in the repository",
                Severity.WARN,
                file_path=found[0]
            )

    else:
        # Fallback to manual detection if ScanCode not available
        if licenses_detected_manual:
            logger.log_pass(
                ProblemCode.LICENSE_TYPE_UNKNOWN,
                f"License type detected (manual): {', '.join(licenses_detected_manual)}",
                licenses=licenses_detected_manual,
                method="manual_pattern_matching"
            )
        else:
            logger.log_problem(
                ProblemCode.LICENSE_TYPE_UNKNOWN,
                "No standard license type detected in LICENSE file",
                Severity.WARN,
                file_path=found[0]
            )


# ============================================================================
# COPYRIGHT CHECKS
# ============================================================================

def copyright_checks(repo_root: Path, logger: ProblemLogger) -> None:
    """COPYRIGHT file content checks."""
    found = find_any(repo_root, ARTEFACTS["COPYRIGHT"])

    if not found:
        logger.log_problem(
            ProblemCode.ARTEFACT_COPYRIGHT_MISSING,
            "COPYRIGHT file not found",
            Severity.SKIP
        )
        return

    cr_path = repo_root / found[0]
    text = read_text_safe(cr_path)
    text_lc = text.lower()

    # Copyright statement with year
    cr_year_ok = bool(re.search(r"copyright\s*(?:\(|©)?\s*(19|20)\d{2}", text_lc))
    if cr_year_ok:
        logger.log_pass(
            ProblemCode.COPYRIGHT_STATEMENT_MISSING,
            "Copyright statement with year detected",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.COPYRIGHT_STATEMENT_MISSING,
            "No clear copyright statement with year in COPYRIGHT file",
            Severity.WARN,
            file_path=found[0]
        )

    # GÉANT mention
    geant_ok = "geant" in text_lc or "géant" in text_lc
    if geant_ok:
        logger.log_pass(
            ProblemCode.COPYRIGHT_GEANT_MISSING,
            "GÉANT copyright mention detected",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.COPYRIGHT_GEANT_MISSING,
            "No GÉANT mention in COPYRIGHT (may not be required for non-GÉANT projects)",
            Severity.WARN,
            file_path=found[0],
            note="Optional for non-GÉANT projects"
        )

    # Multiple copyright holders
    copyright_count = len(re.findall(r"copyright\s*(?:\(|©)?", text_lc))
    if copyright_count >= 1:
        logger.log_pass(
            ProblemCode.COPYRIGHT_STATEMENTS_COUNT,
            f"Found {copyright_count} copyright statement(s)",
            count=copyright_count
        )
    else:
        logger.log_problem(
            ProblemCode.COPYRIGHT_STATEMENTS_COUNT,
            "No copyright statements found",
            Severity.WARN,
            file_path=found[0]
        )

    # EU logo check
    eu_logo_files = ["eu-logo.png", "eu-flag.png", "EU-logo.jpg", "EU-flag.jpg", "eu_flag.svg"]
    eu_logo_ok = any((repo_root / f).exists() for f in eu_logo_files)

    if eu_logo_ok:
        found_logos = [f for f in eu_logo_files if (repo_root / f).exists()]
        logger.log_pass(
            ProblemCode.COPYRIGHT_EU_LOGO_MISSING,
            f"EU logo file detected: {', '.join(found_logos)}",
            logo_files=found_logos
        )
    else:
        logger.log_problem(
            ProblemCode.COPYRIGHT_EU_LOGO_MISSING,
            "No EU logo file detected (eu-logo.png, eu-flag.png, etc.)",
            Severity.WARN,
            expected_locations=eu_logo_files,
            note="Required for EU-funded projects"
        )


# ============================================================================
# AUTHORS CHECKS
# ============================================================================

def authors_checks(repo_root: Path, logger: ProblemLogger) -> None:
    """AUTHORS file content checks."""
    found = find_any(repo_root, ARTEFACTS["AUTHORS"])

    if not found:
        logger.log_problem(
            ProblemCode.ARTEFACT_AUTHORS_MISSING,
            "AUTHORS file not found",
            Severity.SKIP
        )
        return

    auth_path = repo_root / found[0]
    text = read_text_safe(auth_path)
    text_lc = text.lower()

    # GÉANT phase/work package
    geant_phase_ok = bool(re.search(r"gn\d+|géant\s+\d+|work\s+package|wp\d+|task\s+\d+", text_lc))
    if geant_phase_ok:
        logger.log_pass(
            ProblemCode.AUTHORS_GEANT_PHASE_MISSING,
            "GÉANT phase/WP detected",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.AUTHORS_GEANT_PHASE_MISSING,
            "No GÉANT phase/WP detected in AUTHORS (may not be required for non-GÉANT projects)",
            Severity.WARN,
            file_path=found[0]
        )

    # Developers list (heuristic: multiple names/emails)
    email_count = len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text))
    name_lines = len([line for line in text.split("\n") if line.strip() and not line.strip().startswith("#")])
    if email_count >= 1 or name_lines >= 2:
        logger.log_pass(
            ProblemCode.AUTHORS_CONTACTS_MISSING,
            f"Found {email_count} email(s), {name_lines} non-empty lines",
            email_count=email_count,
            name_lines=name_lines
        )
    else:
        logger.log_problem(
            ProblemCode.AUTHORS_CONTACTS_MISSING,
            "Insufficient developer contact information in AUTHORS",
            Severity.WARN,
            file_path=found[0]
        )

    # Contributors mention
    contrib_ok = has_any(text_lc, [r"\bcontributors?\b", r"\bother\b"])
    if contrib_ok:
        logger.log_pass(
            ProblemCode.AUTHORS_CONTRIBUTORS_MISSING,
            "Contributors section detected",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.AUTHORS_CONTRIBUTORS_MISSING,
            "No contributors section in AUTHORS",
            Severity.WARN,
            file_path=found[0]
        )

    # Funding information
    fund_ok = has_any(text_lc, [r"\bfunding\b", r"\bgrant\b", r"\bsponsored\b"])
    if fund_ok:
        logger.log_pass(
            ProblemCode.AUTHORS_FUNDING_MISSING,
            "Funding info detected",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.AUTHORS_FUNDING_MISSING,
            "No funding info in AUTHORS",
            Severity.WARN,
            file_path=found[0]
        )


# ============================================================================
# NOTICE CHECKS
# ============================================================================

def notice_checks(repo_root: Path, repo_name: str, logger: ProblemLogger) -> None:
    """NOTICE file content checks."""
    found = find_any(repo_root, ARTEFACTS["NOTICE"])

    if not found:
        logger.log_problem(
            ProblemCode.ARTEFACT_NOTICE_MISSING,
            "NOTICE file not found",
            Severity.SKIP
        )
        return

    notice_path = repo_root / found[0]
    text = read_text_safe(notice_path)
    text_lc = text.lower()

    # Project name
    proj_ok = (repo_name.lower() in text_lc) or has_any(text_lc, [r"^\s*project\s*[:\-]"])
    if proj_ok:
        logger.log_pass(
            ProblemCode.NOTICE_PROJECT_NAME_MISSING,
            "Project name detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_PROJECT_NAME_MISSING,
            "Project name not detected in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Copyright
    cr_ok = has_any(text_lc, [r"copyright"]) and bool(re.search(r"(19|20)\d{2}", text))
    if cr_ok:
        logger.log_pass(
            ProblemCode.NOTICE_COPYRIGHT_MISSING,
            "Copyright with year detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_COPYRIGHT_MISSING,
            "No copyright with year in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Licence declaration
    lic_ok = has_any(text_lc, [r"licen[cs]e"]) and (has_any(text_lc, [r"https?://", r"\bLICENSE\b", r"\bCOPYING\b"]))
    if lic_ok:
        logger.log_pass(
            ProblemCode.NOTICE_LICENSE_MISSING,
            "Licence declaration detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_LICENSE_MISSING,
            "No licence declaration in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Authors/contributors
    auth_ok = has_any(text_lc, [r"\bauthors?\b", r"\bcontributors?\b", r"\bAUTHORS\b"])
    if auth_ok:
        logger.log_pass(
            ProblemCode.NOTICE_AUTHORS_MISSING,
            "Authors reference detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_AUTHORS_MISSING,
            "No authors reference in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Third-party components
    third_ok = has_any(text_lc, [r"third[-\s]?party", r"dependencies"]) or (text_lc.count("license") >= 2)
    if third_ok:
        logger.log_pass(
            ProblemCode.NOTICE_THIRD_PARTY_MISSING,
            "Third-party section detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_THIRD_PARTY_MISSING,
            "No third-party components listing in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Tools used
    tools_ok = has_any(text_lc, [r"\btools?\b", r"\bbuilt with\b"])
    if tools_ok:
        logger.log_pass(
            ProblemCode.NOTICE_TOOLS_MISSING,
            "Tools mention detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_TOOLS_MISSING,
            "No tools mention in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Trademark
    tm_ok = has_any(text_lc, [r"trademark"]) or ("™" in text) or ("®" in text)
    if tm_ok:
        logger.log_pass(
            ProblemCode.NOTICE_TRADEMARK_MISSING,
            "Trademark wording detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_TRADEMARK_MISSING,
            "No trademark wording in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Patents
    pat_ok = has_any(text_lc, [r"patent"])
    if pat_ok:
        logger.log_pass(
            ProblemCode.NOTICE_PATENTS_MISSING,
            "Patent wording detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_PATENTS_MISSING,
            "No patent wording in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Acknowledgements
    ack_ok = has_any(text_lc, [r"acknowledg", r"\bthanks\b", r"\bfunding\b"])
    if ack_ok:
        logger.log_pass(
            ProblemCode.NOTICE_ACKNOWLEDGEMENTS_MISSING,
            "Acknowledgements detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_ACKNOWLEDGEMENTS_MISSING,
            "No acknowledgements in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )

    # Prior notices
    prior_ok = has_any(text_lc, [r"prior notice", r"this product includes", r"third[-\s]?party notices"])
    if prior_ok:
        logger.log_pass(
            ProblemCode.NOTICE_PRIOR_NOTICES_MISSING,
            "Prior-notice wording detected in NOTICE",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.NOTICE_PRIOR_NOTICES_MISSING,
            "No prior-notice wording in NOTICE",
            Severity.WARN,
            file_path=found[0]
        )


# ============================================================================
# CHANGELOG CHECKS
# ============================================================================

def changelog_checks(repo_root: Path, repo_name: str, logger: ProblemLogger) -> None:
    """CHANGELOG file content checks."""
    found = find_any(repo_root, ARTEFACTS["CHANGELOG"])

    if not found:
        logger.log_problem(
            ProblemCode.ARTEFACT_CHANGELOG_MISSING,
            "CHANGELOG file not found",
            Severity.SKIP
        )
        return

    cl_path = repo_root / found[0]
    text = read_text_safe(cl_path)
    text_lc = text.lower()

    # Project name
    proj_ok = repo_name.lower() in text_lc[:500]
    if proj_ok:
        logger.log_pass(
            ProblemCode.CHANGELOG_PROJECT_NAME_MISSING,
            "Project name detected in CHANGELOG",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.CHANGELOG_PROJECT_NAME_MISSING,
            "Project name not detected in CHANGELOG",
            Severity.WARN,
            file_path=found[0]
        )

    # Reverse chronological order (heuristic: latest version/date first)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    version_lines = [l for l in lines if re.search(r"v?\d+\.\d+(\.\d+)?|unreleased", l, re.IGNORECASE)]
    chrono_ok = len(version_lines) >= 1
    if chrono_ok:
        logger.log_pass(
            ProblemCode.CHANGELOG_NOT_CHRONOLOGICAL,
            "Version entries detected in CHANGELOG",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.CHANGELOG_NOT_CHRONOLOGICAL,
            "No clear version entries in CHANGELOG",
            Severity.WARN,
            file_path=found[0]
        )

    # Version numbers and dates
    version_ok = bool(re.search(r"v?\d+\.\d+(\.\d+)?\s*-?\s*\d{4}", text))
    if version_ok:
        logger.log_pass(
            ProblemCode.CHANGELOG_VERSION_DATE_MISSING,
            "Version + date pattern detected in CHANGELOG",
            file_path=found[0]
        )
    else:
        logger.log_problem(
            ProblemCode.CHANGELOG_VERSION_DATE_MISSING,
            "No clear version/date pattern in CHANGELOG",
            Severity.WARN,
            file_path=found[0]
        )

    # Changelog sections
    sections = [
        (ProblemCode.CHANGELOG_ADDED_MISSING, "Added", [r"^\s*##?\s*added\b", r"^\s*\*\*added\*\*"]),
        (ProblemCode.CHANGELOG_CHANGED_MISSING, "Changed", [r"^\s*##?\s*changed\b", r"^\s*\*\*changed\*\*"]),
        (ProblemCode.CHANGELOG_DEPRECATED_MISSING, "Deprecated",
         [r"^\s*##?\s*deprecated\b", r"^\s*\*\*deprecated\*\*"]),
        (ProblemCode.CHANGELOG_REMOVED_MISSING, "Removed", [r"^\s*##?\s*removed\b", r"^\s*\*\*removed\*\*"]),
        (ProblemCode.CHANGELOG_FIXED_MISSING, "Fixed", [r"^\s*##?\s*fixed\b", r"^\s*\*\*fixed\*\*"]),
        (ProblemCode.CHANGELOG_SECURITY_MISSING, "Security", [r"^\s*##?\s*security\b", r"^\s*\*\*security\*\*"]),
    ]

    for code, name, pats in sections:
        ok = any(re.search(p, text_lc, flags=re.MULTILINE | re.IGNORECASE) for p in pats)
        if ok:
            logger.log_pass(code, f"{name} section detected in CHANGELOG", file_path=found[0])
        else:
            logger.log_problem(code, f"No {name} section in CHANGELOG", Severity.WARN, file_path=found[0])


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python check_repo_artefacts.py <git_url> [--format csv|json|both]")
        print("Example: python check_repo_artefacts.py https://github.com/GEANT/CAT.git --format both")
        return 2

    git_url = sys.argv[1].strip()
    output_format = "csv"  # default

    if len(sys.argv) >= 4 and sys.argv[2] == "--format":
        output_format = sys.argv[3].lower()
        if output_format not in ["csv", "json", "both"]:
            print("ERROR: format must be csv, json, or both")
            return 2

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

        # Initialize problem logger
        logger = ProblemLogger(git_url)

        # ===== SCANCODE LICENSE DETECTION =====
        scancode_data = None
        if check_scancode_available():
            print("ScanCode is available, running license detection...")
            scancode_data = run_scancode_license_scan(repo_dir)
            if scancode_data:
                print("ScanCode scan completed successfully")
            else:
                logger.log_problem(
                    ProblemCode.LICENSE_SCANCODE_FAILED,
                    "ScanCode scan failed or timed out",
                    Severity.INFO,
                    hint="License detection will fall back to pattern matching"
                )
        else:
            logger.log_problem(
                ProblemCode.LICENSE_SCANCODE_NOT_AVAILABLE,
                "ScanCode toolkit not available - install with: pip install scancode-toolkit",
                Severity.INFO,
                hint="License detection will use basic pattern matching only"
            )
            print("ScanCode not available - using basic pattern matching for license detection")
            print("Install ScanCode for more accurate license detection: pip install scancode-toolkit")

        # ===== KEY ARTEFACTS PRESENCE =====
        print("Checking key artifacts...")
        for artefact, candidates in ARTEFACTS.items():
            found = find_any(repo_dir, candidates)
            code_name = f"ARTEFACT_{artefact}_MISSING"
            code = ProblemCode[code_name]

            if found:
                logger.log_pass(code, f"{artefact} file found: {', '.join(found)}", files=found)
            else:
                logger.log_problem(
                    code,
                    f"{artefact} file not found",
                    Severity.FAIL if artefact in ["README", "LICENSE"] else Severity.WARN
                )

        # ===== BINARY FILES =====
        print("Scanning for binary files...")
        scan_binaries(repo_dir, logger)

        # ===== LARGE FILES =====
        print("Scanning for large files...")
        scan_large_files(repo_dir, LARGE_FILE_THRESHOLD_BYTES, logger)

        # ===== CONTENT CHECKS =====
        print("Analyzing README content...")
        readme_checks(repo_dir, repo_name, logger)

        print("Analyzing LICENSE content...")
        license_checks(repo_dir, logger, scancode_data)

        print("Analyzing COPYRIGHT content...")
        copyright_checks(repo_dir, logger)

        print("Analyzing AUTHORS content...")
        authors_checks(repo_dir, logger)

        print("Analyzing NOTICE content...")
        notice_checks(repo_dir, repo_name, logger)

        print("Analyzing CHANGELOG content...")
        changelog_checks(repo_dir, repo_name, logger)

        # ===== CONDITIONAL CHECKS =====
        # Contributing fallback
        found_contrib = find_any(repo_dir, ARTEFACTS["CONTRIBUTING"])
        if not found_contrib:
            rp = readme_path(repo_dir)
            if rp:
                md = read_text_safe(rp).lower()
                contrib_ok = bool(re.search(r"\bcontributing\b", md))
                if contrib_ok:
                    logger.log_pass(
                        ProblemCode.CONTRIBUTING_FALLBACK_MISSING,
                        "Contributing found in README (fallback)",
                        file_path="README"
                    )
                else:
                    logger.log_problem(
                        ProblemCode.CONTRIBUTING_FALLBACK_MISSING,
                        "Not in CONTRIBUTING file or README",
                        Severity.FAIL
                    )
            else:
                logger.log_problem(
                    ProblemCode.CONTRIBUTING_FALLBACK_MISSING,
                    "No CONTRIBUTING file and README missing",
                    Severity.FAIL
                )

        # Authors fallback
        found_authors = find_any(repo_dir, ARTEFACTS["AUTHORS"])
        if not found_authors:
            rp = readme_path(repo_dir)
            if rp:
                md = read_text_safe(rp).lower()
                authors_ok = bool(re.search(r"\bauthor|contributors?\b", md))
                if authors_ok:
                    logger.log_pass(
                        ProblemCode.AUTHORS_FALLBACK_MISSING,
                        "Authors found in README (fallback)",
                        file_path="README"
                    )
                else:
                    logger.log_problem(
                        ProblemCode.AUTHORS_FALLBACK_MISSING,
                        "Not in AUTHORS file or README",
                        Severity.WARN
                    )

        # ===== WRITE RESULTS =====
        if output_format in ["csv", "both"]:
            csv_file = Path("artefact_check.csv")
            logger.to_csv(csv_file)
            print(f"\n{'=' * 70}")
            print(f"CSV report written to: {csv_file.resolve()}")

        if output_format in ["json", "both"]:
            json_file = Path("artefact_check.json")
            logger.to_json(json_file)
            print(f"JSON report written to: {json_file.resolve()}")

        # Print summary
        summary = logger.get_summary()
        print(f"\n{'=' * 70}")
        print(f"Total checks performed: {summary['total']}")
        print(f"\nResults Summary:")
        print(f"  PASS: {summary['pass']}")
        print(f"  INFO: {summary['info']}")
        print(f"  WARN: {summary['warn']}")
        print(f"  FAIL: {summary['fail']}")
        print(f"  SKIP: {summary['skip']}")

        # Print high-priority failures
        high_priority_fails = [p for p in logger.problems if p.severity == Severity.FAIL and p.priority <= 2]
        if high_priority_fails:
            print(f"\n{'=' * 70}")
            print("HIGH PRIORITY ISSUES:")
            for p in high_priority_fails:
                print(f"  [{p.code.value}] {p.message}")

        print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())