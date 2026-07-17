"""Shared test fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add src to path so we can import agent package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository directory with sample Java files."""
    src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
    src_dir.mkdir(parents=True)
    test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
    test_dir.mkdir(parents=True)

    # Sample Java source
    (src_dir / "Hello.java").write_text(
        'package com.example;\n\npublic class Hello {\n'
        '    public String greet(String name) {\n'
        '        return "Hello, " + name;\n'
        '    }\n}\n',
        encoding="utf-8",
    )

    # Sample test
    (test_dir / "HelloTest.java").write_text(
        'package com.example;\n\nimport org.junit.jupiter.api.Test;\n'
        "import static org.junit.jupiter.api.Assertions.*;\n\n"
        "public class HelloTest {\n"
        "    @Test\n"
        "    void greet() {\n"
        '        Hello h = new Hello();\n'
        '        assertEquals("Hello, World", h.greet("World"));\n'
        "    }\n}\n",
        encoding="utf-8",
    )

    return tmp_path
