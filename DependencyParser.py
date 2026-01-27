#!/usr/bin/env python3
"""
Copyright (c) 2025 GÉANT Association on behalf of the GÉANT project
DependencyParser is licensed under the MIT License.
See https://opensource.org/licenses/MIT for the full licence text.

The [GÉANT project](https://geant.org/projects/) is funded by the Horizon Europe research and innovation programme.
DependencyParser has been created by the GÉANT 5-2 WP9 T2 Software Licence Management.

Developers:
- Branko Marović: Initial design and programming

This utility parses markdown text to extract information about software dependencies listed under specific sections.
It automates the extraction of dependency licensing and version information from markdown files to support compliance and inventory processes.

The tool parses README.md, README, NOTICE.md, and NOTICE files. From markdown text, it extracts information about software dependencies listed under specific sections.

Recognised section titles (marked with '#', '##', etc.) include "Dependencies", "Libraries", "Third-Party Components", "Third Party Components", "Third-Party Libraries", "Third Party Libraries", "External Dependencies", and "External Libraries".

A section ends with another section or the end of the file. Section titles and tag names within sections are case-insensitive.

Within each section, it identifies individual dependency entries starting with standard markdown list markers ('-', '*', or '1.').

For each dependency, it extracts the following metadata fields (optional fields default to None when missing):
- from_file: Source file name (automatically added)
- name: Dependency name (required)
- name_url: Project URL (optional)
- version: Version string (optional)
- description: Description text (optional)
- license: Licence type name, as provided (optional)
- license_line: Source line number where licence appears (optional)
- copyright_text: Copyright notice (optional)
- copyright_line: Source line number where copyright appears (optional)

To facilitate matching with ScanCode JSON output, it records line numbers where the licence name and copyright appear.

Within sections, it supports flexible markdown formatting, including optional bold ('**'), and variations in separators (colons, semicolons, commas, '-' and '–' dashes).

The tool outputs all extracted dependencies as JSON, including the collected fields and line references.

Supported Dependency Block Formats

Compact bulleted or numbered list item:
- Starts with a name (with optional link), followed by an optional version, licence in parentheses (if present), and description.
- A comma, colon, or dash is used as a separator where needed.
- If the name is without a link, a separator must divide it from the version.
- An optional separator follows.
- The optional licence (in parentheses) may be followed by an optional separator and then an optional description.
- If there is no licence, a separator must divide the version from the description.

Block format:
- The first line follows the compact format described above.
- Additional indented lines (by at least 2 spaces) provide supplementary information.
- These lines may contain tagged fields (e.g., 'Licence: MIT'), except for the copyright line, which remains untagged.
- Indented lines may, but do not have to be, bulleted. Blank lines terminate the block.
- The lines after the first contain tagged fields providing information not included in the first line.

Parsing Rules and Flexibility

- Blank or plain text lines are allowed between dependencies within a section.
- Both 'Licence' and 'License' spellings are accepted.
- Leading '-', '*', or numbered '1.' formats are supported for list items.
- Colons, commas, '-' and '–' dashes are valid separators.
- Markdown decoration (like '**') is optional and ignored during parsing.
- The name field is required; all other fields are optional.
- Optional fields (name_url, version, description, license, copyright_text) default to None when not found.
- When licence is not found, both license and license_line are None.
- When copyright is not found, both copyright_text and copyright_line are None.
- Malformed version strings (e.g., v1.2-alpha vs 1.2.0) are accepted without validation.
- Relative URLs (e.g., [lib](./path)) are allowed.
- Duplicate dependencies (including the same name with different versions) are permitted.

Tested at https://www.online-python.com/

Potential future additions:
- Image field handling, e.g.:
  Image: dockerhub.io/dep:latest
- Validation of extracted URLs and version formats
- Multi-line copyright notices (until the next section, empty line, or higher-level list item), e.g.:
  Copyright (c) 2020-2024
  John Doe and Contributors
  All rights reserved

This licence pattern detection may be needed later if we start supporting less deterministic formats. It currently includes only keywords for most common licences:
        self.license_keywords = re.compile(
            r'\b('
            r'afl|academic\s+free|'
            r'agpl|affero\s+general|'
            r'antlr|'
            r'apache|'
            r'artistic|'
            r'bouncy\s*castle|'
            r'bsd|berkeley\s+software|'
            r'bsl|boost|'
            r'bzip2|'
            r'cc-by|cc\s+by|'
            r'cc0|wtfpl|unlicense|'
            r'cddl|'
            r'cecill|'
            r'creative\s+commons|'
            r'epl|'
            r'eupl|european\s+union\s+public|'
            r'ftl|freetype|'
            r'futc|free\s+use|'
            r'gpl|gnu|'
            r'icu|'
            r'imagemagick|'
            r'isc|0bsd|'
            r'json|'
            r'ijg|independent\s+jpeg\s+group|'
            r'lgpl|'
            r'libpng|'
            r'mit|x11|massachusetts\s+institute|'
            r'mpl|mozilla|'
            r'ncsa|university\s+of\s+illinois|'
            r'nunit|'
            r'open\s*ldap|'
            r'openssl|'
            r'osl|open\s+software|'
            r'postgre\s*sql|'
            r'public\s+domain|'
            r'python|'
            r'ruby|'
            r'sleepycat|'
            r'sspl|server\s+side\s+public|'
            r'zlib|'
            r'zpl|zope'
            r')\b',
            re.IGNORECASE
        )
"""
import re
import json
import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple, Any

# Test string for validation
test_string = """
# Project Documentation
This is a comprehensive test project for the dependency parser.
## Installation
Run pip install requirements.txt to install dependencies.
# Dependencies
This section contains various dependency formats to test the parser comprehensively.
Compact Format Tests
- **[TensorFlowX](https://tensorflowx.org)**
Empty line
-
- **[PyTorchY](https://pytorchy.dev)**version 1.9.2
1. **[KerasZ](https://kerasz.io)**0.5.7
- **[MXNetA](https://mxnet.pro)** version 3.1.4 (Apache-2.0)
- [CaffeB](https://caffeb.net) : version 2.3.1 : (BSD-3-Clause) - Deep learning framework
- [TheanoC](https://theanoc.ai) : 1.0.5 : (MIT) Numerical computation library
* [ONNXD](https://onnxd.org), 1.12.0 (MIT) Neural network exchange format
- [CNTKE](https://cntke.io) version 2.8.3 - Cognitive toolkit
* **[FastAIF](https://fastaif.dev)** (Apache-2.0): Deep learning library
- [ChainerG](https://chainer.gq) (MIT) : Neural network framework
1. [PaddleH](https://paddleh.ai) (Apache-2.0)
- **[GeoData W](https://geodata.org)**: Geospatial analysis toolkit
- DLibI: Machine learning utilities
-  OpenCVJ, version 4.7.0 (BSD-3-Clause)  - Computer vision library
* TorchK (BSD-3-Clause): Scientific computing
* NeuralNetX: Deep learning framework
* QuantumCoreY, version 3.5.2, Quantum algorithm library
1. **BioML Z** - v2.7.1 (Apache-2.0)
- **DeepVision** - Computer vision toolkit
- SciKitL: v1.3.0 (BSD-3-Clause)
* CatBoostM (Apache-2.0), Gradient boosting library
- XGBoostN - (Apache-2.0) - Optimized distributed gradient boosting
Indented Block Format Tests
- **RayO**
  - Version: 2.6.1
  - **Image:** docker.io/rayproject/ray:2.6.1
  - **Licence:** Apache-2.0
  - **URL:** https://ray.io
  - **Description:** Distributed computing framework
  - Copyright (c) 2023 Ray Project
- HorovodP
  - © Copyright 2023 Horovod Team
  Version: 0.28.1
  Description: Distributed training framework
  License: Apache-2.0
  URL: https://horovod.ai
- [DaskQ](https://daskq.org) 2023.9.1 - Parallel computing
  Description: Flexible parallel computing
  License: BSD-3-Clause
1. **[ModinR](https://modinr.readthedocs.io)**
   Copyright © 2023-2025 Modin Developers
   Licence: Apache-2.0
- VaexS 4.13.0 (MIT License)
      Description: Out-of-core DataFrames
      - **Version:** 4.13.0
      - Copyright 2023-2025 Vaex Team
Edge Cases
- **[Mixed Library](https://example.com/mixed)** - Version 5.0.0, MIT License
  Copyright (c) 2023 Mixed Corp
* Plain Library Name - Apache-2.0 License
  - Version: 1.1.1
  - Copyright 2022 Plain Corp
1. **[Versionless Lib](https://github.com/user/versionless)** - GPL-2.0
   Copyright (c) 2021 Versionless Inc
- Minimal Lib - MIT
* Just Name Library
- **Docker Component**
  - **Version:** 1.2.3
  - **Image:** dockerhub.io/dep:latest
  - **License:** MIT
  - **URL:** https://example.com/dep
  - Copyright (c) 2020 Docker Example
# Third-Party Components
Additional components used in this project.
- **[Bootstrap](https://getbootstrap.com)** - Version 5.3.0
  - License: MIT
  - Copyright (c) 2011-2023 The Bootstrap Authors
* **[jQuery](https://jquery.com)** - Version 3.7.0
  - Licence: MIT
  - Copyright © 2023 OpenJS Foundation and other contributors
1. **Moment.js**
   - Version: 2.29.4
   - License: MIT
   - URL: https://momentjs.com
   - Copyright (c) JS Foundation and other contributors
- **Chart.js**
  - Version: 4.3.0
  - License: MIT
  - URL: https://www.chartjs.org
  - Copyright (c) 2014-2023 Chart.js Contributors
# Third Party Libraries
Testing alternative section title format.
- **[Lodash](https://lodash.com)** - Version 4.17.21
  License: MIT
  Copyright © 2012-2023 The Dojo Foundation
* **[Underscore.js](https://underscorejs.org)** version 1.13.6 (MIT License) - Utility library
# External Dependencies
Build Tools
1. **Webpack** - Run webpack on the command-line to create bundle.js
   - Version: 5.88.0
   - Licence: MIT
   - URL: https://webpack.js.org
   - Copyright JS Foundation and other contributors
* **Babel**
  - Version: 7.22.5
  - License: MIT
  - URL: https://babeljs.io
  - Copyright (c) 2014-present Sebastian McKenzie
# libraries
Testing lowercase section title (case-insensitive).
- **Lowercase Test Lib** - MIT License
  Version: 1.0.0
  Copyright (c) 2023 Test Corp
# External Libraries
Final section title test.
- **[TypeScript](https://typescriptlang.org)** version 5.1.6 (Apache-2.0) - Static type checker
* Sass, v1.63.6 (MIT License): CSS preprocessor
## Documentation
This should not be parsed as it's not under a recognized dependency section.
- Some Random Library - MIT License
## Other Random Section
More content that should be ignored.
# Configuration
Project configuration details go here.
## Dependencies for Testing
This section title contains "Dependencies" but is not exactly "Dependencies", so should be ignored.
- **Dev Library** - MIT License
"""

class DependencyParser:
    """Main parser class for extracting dependency information from Markdown files.

    This class handles all aspects of parsing:
    - Section detection
    - Dependency block identification
    - Field extraction
    - Data cleaning and normalization

    The parser uses regular expressions for pattern matching and maintains state
    about the current parsing context.
    """

    def section_titles(self) -> List[str]:
        """Return the list of recognized section titles that may contain dependencies.

        Returns:
            List of strings representing valid section titles that may contain
            dependency information. These are matched case-insensitively.
        """
        return [
            "Dependencies",
            "Libraries",
            "Third-Party Components",
            "Third Party Components",
            "Third-Party Libraries",
            "Third Party Libraries",
            "External Dependencies",
            "External Libraries"
        ]

    def __init__(self):
        """Initialize the parser with compiled regex patterns for efficient matching.

        Compiles all regular expressions used throughout the parsing process.
        These patterns are compiled once at initialization for better performance.
        """
        # Section header pattern - matches recognized section titles at any heading level
        self.section_pattern = re.compile(
            rf'^(#{{1,6}})\s*({"|".join(re.escape(title) for title in self.section_titles())})\s*$',
            re.IGNORECASE | re.MULTILINE
        )

        # Pattern to identify the start of any section
        self.next_section_pattern = re.compile(r'^#{1,6}\s+\S+', re.MULTILINE)

        # Pattern to match list item markers (-, *, or numbered)
        self.list_marker_pattern = re.compile(r'^(\s*)([-*]|\d+\.)\s+')

        # Pattern to match bold Markdown formatting
        self.bold_pattern = re.compile(r'\*{2}([^*]+)\*{2}')

        # Pattern to match Markdown links [text](url)
        self.markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        # Collection of patterns to identify copyright statements
        self.copyright_patterns = [
            # Core copyright/copyleft indicators
            re.compile(r'\b(?:copy(?:right|left))\b|[©🄯Ⓒↄↄ]|\(c\)', re.IGNORECASE), # "copyright", "copyleft", ©, Ⓒ, (c), 🄯,,ↄ

            # Standard legal restrictions
            re.compile(r'\ball\s+rights\s+reserved\b', re.IGNORECASE),
            re.compile(r'\bproprietary\s+and\s+confidential\b', re.IGNORECASE), # Proprietary/confidential notice
            re.compile(r'\bnot\s+for\s+distribution\b', re.IGNORECASE), # Restriction on sharing
            re.compile(r'\bfor\s+internal\s+use\s+only\b', re.IGNORECASE), # Use restriction
            re.compile(r'\brestricted\s+rights\s+legend\b', re.IGNORECASE), # Common in US gov contracts
            re.compile(r'\bunauthorised\s+copying\s+prohibited\b', re.IGNORECASE), # Explicit restriction
            re.compile(r'\bcommercial\s+licen[cs]e\s+required\b', re.IGNORECASE), # Non-free licence required
            re.compile(r'\blicensed\s+only\s+for\s+use\s+with\b', re.IGNORECASE), # Tied licence
            re.compile(r'\bmay\s+not\s+be\s+reproduced\s+or\s+used\s+without\s+permission\b', re.IGNORECASE),
            re.compile(r'\bownership\s+of\s+this\s+software\s+remains\s+with\b', re.IGNORECASE),

            # Abbreviations and international variants
            re.compile(r'\b(copr|copyr)\.?\b', re.IGNORECASE), # Abbreviated copyright
            re.compile(r'\bderechos\s+reservados\b', re.IGNORECASE), # Spanish
            re.compile(r'\burheberrecht\b', re.IGNORECASE), # German
            re.compile(r'著作権', re.IGNORECASE), # Japanese
            re.compile(r'版權', re.IGNORECASE), # Traditional Chinese
            re.compile(r'авторские\s+права', re.IGNORECASE), # Russian
            re.compile(r'izquierdos\s+autorizados', re.IGNORECASE), # Satirical Spanish copyleft
            re.compile(r'copie\s+gauche', re.IGNORECASE), # French
            re.compile(r'コピーレフト', re.IGNORECASE), # Japanese copyleft
            re.compile(r'著佐权', re.IGNORECASE), # Chinese copyleft
        ]

        # Patterns for extracting specific metadata fields
        self.version_pattern = re.compile(r'\*{0,2}version\*{0,2}\s*:\s*(.+)', re.IGNORECASE)
        self.url_pattern = re.compile(r'\*{0,2}url\*{0,2}\s*:\s*(.+)', re.IGNORECASE)
        self.license_pattern = re.compile(r'\*{0,2}licen[cs]e\*{0,2}\s*:\s*(.+)', re.IGNORECASE)
        self.description_pattern = re.compile(r'\*{0,2}description\*{0,2}\s*:\s*(.+)', re.IGNORECASE)

    def extract_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """Extract all relevant sections from the Markdown text.

        Args:
            text: The complete Markdown text to parse

        Returns:
            List of tuples containing:
            - section title
            - section content
            - starting position in the text
        """
        matches = list(self.section_pattern.finditer(text))
        #print(f"Found {len(matches)} section matches")  # Debug print
        sections = []

        for i, match in enumerate(matches):
            #print(f"Matched section: {match.group(2)}")  # Debug print
            start = match.end()
            # Find next section or end of file
            next_match = self.next_section_pattern.search(text, start)
            end = next_match.start() if next_match else len(text)
            sections.append((match.group(2).strip(), text[start:end], match.start()))

        return sections

    def is_copyright_line(self, line: str) -> bool:
        """Determine if a line contains copyright information.

        Args:
            line: The text line to check

        Returns:
            True if the line matches any copyright pattern, False otherwise
        """
        return any(pattern.search(line) for pattern in self.copyright_patterns)

    def extract_markdown_link(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract name and URL from a Markdown link format.

        Args:
            text: Text potentially containing a Markdown link

        Returns:
            Tuple of (link_text, link_url) if found, (None, None) otherwise
        """
        match = self.markdown_link_pattern.search(text.strip())
        return (match.group(1).strip(), match.group(2).strip()) if match else (None, None)

    def clean_license_text(self, text: str) -> str:
        """Normalize and clean license text by removing extraneous formatting.

        Args:
            text: Raw license text from source

        Returns:
            Cleaned and normalized license text
        """
        if not text:
            return text

        # Handle markdown links in license
        link_match = self.markdown_link_pattern.search(text.strip())
        if link_match:
            print(f"link_match: '{link_match.group(1).strip()}'")  # Debug print
            return link_match.group(1).strip()

        # Remove parentheses around license
        text = re.sub(r'^\s*\(\s*(.+?)\s*\)\s*$', r'\1', text.strip())
        return text.strip()

    def parse_inline_entry(self, line: str, line_number: int) -> Dict[str, Any]:
        """Parse a single-line dependency declaration.

        Handles compact format dependency declarations that fit on one line.

        Args:
            line: The line of text to parse
            line_number: Original line number for reference

        Returns:
            Dictionary containing parsed fields from the line
        """
        clean_line = self.list_marker_pattern.sub('', line.strip())

        # Initialize all fields explicitly (your preferred style)
        name = None
        name_url = None
        version = None
        license_text = None
        license_line = None
        description = None
        remaining_text = clean_line

        # Step 1: Extract name (with optional URL/bold formatting)
        # Handle bold formatting
        bold_match = self.bold_pattern.search(remaining_text)
        if bold_match:
            bold_content = bold_match.group(1)
            link_name, link_url = self.extract_markdown_link(bold_content)
            name = link_name or bold_content
            name_url = link_url
            remaining_text = self.bold_pattern.sub('', remaining_text, count=1).strip()
        else:
            # Handle non-bold links
            link_name, link_url = self.extract_markdown_link(remaining_text)
            if link_name:
                name = link_name
                name_url = link_url
                remaining_text = self.markdown_link_pattern.sub('', remaining_text, count=1).strip()

        # Step 2: Non-link name - everything up to first separator or (
        if not name:
            # Find first separator
            separator_match = re.search(r'^([^(,:;]*?)(?:\s*([(,:;]| -| –)\s*(.*))?$', remaining_text)
            if separator_match:
                name = separator_match.group(1).strip()
                remaining_text = separator_match.group(3) or ""
            else:
                name = remaining_text
                remaining_text = ""

        # Step 3: Process remaining text for version, license, and description
        if remaining_text:
            # First, extract any license in parentheses
            paren_match = re.search(r'^(.*?)\s*\(([^)]+)\)\s*(.*)$', remaining_text)
            if paren_match:
                before_paren = paren_match.group(1).strip()
                # License is what's in parentheses
                license_text = paren_match.group(2).strip()
                after_paren = paren_match.group(3).strip()
                license_line = line_number

                # Version is what comes before parentheses (if anything)
                if before_paren:
                    version = before_paren

                # Description is what comes after parentheses
                if after_paren:
                    description = re.sub(r'^[-–:;,]\s*', '', after_paren).strip()
            else:
                # No parentheses, so version is everything until next separator (if any)
                version_end = len(remaining_text)
                for sep in [' -', ' –', ',', ':', ';']:
                    idx = remaining_text.find(sep)
                    if 0 < idx < version_end:
                        version_end = idx

                if version_end < len(remaining_text):
                    version = remaining_text[:version_end].strip()
                    description = re.sub(r'^[-–:;,]\s*', '', remaining_text[version_end:]).strip()
                else:
                    version = remaining_text.strip()

        # Clean up empty values
        if version and not version:
            version = None
        if description and not description:
            description = None

        # Build final result (consistent with existing return format)
        return {
            'name': name,
            'version': version,
            'name_url': name_url,
            'description': description,
            'license': license_text,
            'license_line': license_line
        }

    def process_text(self, text: str, from_file: str = "<test_string>") -> List[Dict[str, Any]]:
        """Process Markdown text to extract dependencies.

        Args:
            text: The Markdown text to process
            from_file: Source file name for reference

        Returns:
            List of dependency dictionaries found in the text
        """
        sections = self.extract_sections(text)
        all_deps = []

        for section_title, section_text, section_start in sections:
            # Calculate line offset
            line_offset = text[:section_start].count('\n')
            deps = self.parse_dependencies(section_text, line_offset, from_file)
            all_deps.extend(deps)

        return all_deps

    def parse_dependency_block(self, block: List[str], start_line: int, from_file: str) -> Optional[Dict[str, Any]]:
        """Parse a complete dependency block (may span multiple lines).

        Args:
            block: List of lines comprising the dependency declaration
            start_line: Starting line number in original file
            from_file: Source file name for reference

        Returns:
            Dictionary containing all extracted dependency information,
            or None if the block doesn't contain valid dependency data
        """
        if not block:
            return None

        name = None
        name_url = None
        version = None
        license_text = None
        license_line = None
        description = None
        copyright_text = None
        copyright_line = None

        # Process each line in the block
        for idx, line in enumerate(block):
            line_number = start_line + idx
            original_line = line
            clean_line = line.strip()

            # Remove list markers but preserve content for copyright detection
            list_marker_match = self.list_marker_pattern.match(clean_line)
            if list_marker_match:
                clean_line = self.list_marker_pattern.sub('', clean_line)

            # Check for copyright first (before removing other formatting)
            if self.is_copyright_line(original_line):
                copyright_text = original_line.strip()
                # Strip leading Markdown list marker for copyright
                if list_marker_match:
                    copyright_text = self.list_marker_pattern.sub('', copyright_text)
                copyright_line = line_number
                continue

            # If this is the first line, try to extract the name and inline info
            if idx == 0:
                inline_data = self.parse_inline_entry(original_line, line_number)
                name = inline_data['name']
                name_url = inline_data['name_url'] or name_url
                version = inline_data['version'] or version
                if inline_data['license']:
                    license_text = inline_data['license']
                    license_line = inline_data['license_line']
                description = inline_data['description'] or description
                continue

            # Check for structured fields in subsequent lines
            url_match = self.url_pattern.match(clean_line)
            if url_match:
                name_url = url_match.group(1).strip()
                continue

            version_match = self.version_pattern.match(clean_line)
            if version_match:
                version = version_match.group(1).strip()
                continue

            license_match = self.license_pattern.match(clean_line)
            if license_match:
                license_text = self.clean_license_text(license_match.group(1))
                license_line = line_number
                continue

            description_match = self.description_pattern.match(clean_line)
            if description_match:
                description = description_match.group(1).strip()
                continue

        # Final cleanup and validation
        if name:
            # Remove any remaining Markdown formatting
            name = self.markdown_link_pattern.sub(r'\1', name).strip()
            name = self.bold_pattern.sub(r'\1', name).strip()

        # Don't return entries without a name
        if not name or not name.strip():
            return None

        return {
            "from_file": from_file,
            "name": name.strip(),
            "version": version.strip() if version else None,
            "name_url": name_url.strip() if name_url else None,
            "description": description.strip() if description else None,
            "license": license_text.strip() if license_text else None,
            "license_line": license_line,
            "copyright_text": copyright_text.strip() if copyright_text else None,
            "copyright_line": copyright_line
        }

    def parse_dependencies(self, section_text: str, offset_line: int, from_file: str) -> List[Dict[str, Any]]:
        """Parse all dependencies from a section's text content.

        Args:
            section_text: The content of a recognized section
            offset_line: Line number offset from start of file
            from_file: Source file name for reference

        Returns:
            List of parsed dependency dictionaries
        """
        if not section_text.strip():
            return []

        #print(f"Processing section text: {section_text[:100]}...")  # Debug print
        deps = []
        lines = section_text.strip().splitlines()
        i = 0

        while i < len(lines):
            line = lines[i]
            # Check if this is a dependency entry (starts with -, *, or number.)
            first_line_match = self.list_marker_pattern.match(line)
            if first_line_match:
                #print(f"Found dependency at line {i}: {line[:50]}...")  # Debug print
                base_indent = len(first_line_match.group(1))
                block = [line]
                current_line_num = offset_line + i + 1
                i += 1

                # Collect continuation lines that are indented by at least 2 spaces more than base
                while i < len(lines):
                    raw_line = lines[i]
                    next_line = raw_line.rstrip('\n')

                    # Check for new block start or section
                    next_list_match = self.list_marker_pattern.match(next_line)
                    if (next_list_match and
                        len(next_list_match.group(1)) <= base_indent):
                        break

                    if self.next_section_pattern.match(next_line):
                        break

                    # Check for blank line followed by new list item
                    if (not next_line.strip() and i + 1 < len(lines) and
                        self.list_marker_pattern.match(lines[i + 1].strip())):
                        break

                    # Check indentation for continuation lines
                    indent_match = re.match(r'^(\s*)\S', next_line)
                    if not indent_match:
                        # Empty line - could be part of block or terminator
                        if not next_line.strip():
                            i += 1
                            continue
                        break

                    next_indent = len(indent_match.group(1))
                    # Lines indented 2 or more spaces more than the first line are part of the block
                    if (next_indent - base_indent) >= 2:
                        block.append(raw_line)
                        i += 1
                    else:
                        break

                # Parse the collected block
                dep = self.parse_dependency_block(block, current_line_num, from_file)
                if dep:
                    deps.append(dep)
            else:
                i += 1

        return deps

    def process_file(self, from_file: str) -> List[Dict[str, Any]]:
        """Process a single Markdown file.

        Args:
            from_file: Source file name for reference

        Returns:
            List of dependency dictionaries
        """
        if not os.path.exists(from_file):
            return []

        try:
            with open(from_file, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return []  # Silently skip missing files
        except UnicodeDecodeError:
            print(f"Warning: Could not decode {from_file} as UTF-8 - skipping", file=sys.stderr)
            return []
        except Exception as e:
            print(f"Error reading {from_file}: {str(e)}", file=sys.stderr)
            return []

        return self.process_text(text, from_file)

def main(use_test_string: bool = False, output_file: Optional[str] = None) -> None:
    """Main entry point for the dependency parser.

    Handles command line arguments and orchestrates the parsing process.

    Args:
        use_test_string: If True, uses built-in test data instead of files
        output_file: Optional path to write JSON results
    """
    parser = DependencyParser()

    if use_test_string:
        all_dependencies = parser.process_text(test_string)
    else:
        files_to_process = ["README.md", "README", "NOTICE.md", "NOTICE"]
        all_dependencies = []
        for from_file in files_to_process:
            deps = parser.process_file(from_file)
            all_dependencies.extend(deps)

    # Convert results to JSON
    results = json.dumps(all_dependencies, indent=2)

    # Write to file or print to stdout
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(results)
            print(f"Results successfully written to {output_file}")
        except IOError as e:
            print(f"Error writing to {output_file}: {e}", file=sys.stderr)
            print("Falling back to stdout output...", file=sys.stderr)
            print(results)
    else:
        print(results)

if __name__ == "__main__":
    # Set up argument parsing with detailed descriptions
    parser = argparse.ArgumentParser(
        description='Parse software dependencies from Markdown documentation files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  Basic usage (process README/NOTICE files in current directory):
    %(prog)s
  Use built-in test data:
    %(prog)s --test
  Save results to file:
    %(prog)s --output dependencies.json
  Process test data and save to file:
    %(prog)s --test --output test_dependencies.json

Output Format:
  Returns a JSON array of dependency objects. Each dependency includes:
  - from_file: Source file name (automatically added)
  - name: Dependency name (required)
  - name_url: Project URL if provided (optional, None if missing)
  - version: Version string if provided (optional, None if missing)
  - description: Description text if provided (optional, None if missing)
  - license: Licence type if provided (optional, None if missing)
  - license_line: Line number where licence appears (optional, None if missing)
  - copyright_text: Copyright notice if provided (optional, None if missing)
  - copyright_line: Line number where copyright appears (optional, None if missing)
""")

    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='use built-in test data instead of scanning files'
    )
    parser.add_argument(
        '--output', '-o',
        metavar='FILE',
        type=str,
        help='write results to specified JSON file (default: stdout)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0',
        help='show program version and exit'
    )

    args = parser.parse_args()
    main(use_test_string=args.test, output_file=args.output)