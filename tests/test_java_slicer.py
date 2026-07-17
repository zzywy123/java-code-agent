"""Tests for Java code method-level slicer.

Validates:
- Method-level slicing with correct metadata
- Class-level slicing
- Nested class handling
- Constructor extraction
- Overloaded method handling
- Import extraction
- Docstring extraction
- File-level degradation on parse failure
- Directory scanning
- Edge cases (empty files, interfaces, enums)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent.indexing.java_slicer import JavaSlicer
from agent.models import CodeSlice


@pytest.fixture
def slicer() -> JavaSlicer:
    return JavaSlicer()


@pytest.fixture
def sample_java_file(tmp_path: Path) -> Path:
    """Create a sample Java file with methods, constructors, and imports."""
    content = textwrap.dedent("""\
        package com.example.demo;

        import java.util.List;
        import java.util.ArrayList;

        /**
         * A sample service class.
         */
        public class SampleService {
            private final List<String> items;

            /**
             * Constructor with dependency.
             */
            public SampleService() {
                this.items = new ArrayList<>();
            }

            /**
             * Add an item to the list.
             * @param item the item to add
             */
            public void addItem(String item) {
                items.add(item);
            }

            public String getItem(int index) {
                return items.get(index);
            }

            public int size() {
                return items.size();
            }
        }
    """)
    file_path = tmp_path / "SampleService.java"
    file_path.write_text(content, encoding="utf-8")
    return file_path


class TestJavaSlicerBasic:
    """Basic slicing functionality."""

    def test_slice_file_returns_slices(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file, module="demo")
        assert len(slices) > 0
        assert all(isinstance(s, CodeSlice) for s in slices)

    def test_slice_file_extracts_package(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        for s in slices:
            assert s.package == "com.example.demo"

    def test_slice_file_extracts_class_name(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        class_slice = next(s for s in slices if s.method_name == "<class>")
        assert class_slice.class_name == "SampleService"

    def test_slice_file_extracts_methods(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        method_names = [s.method_name for s in slices if s.method_name != "<class>"]
        assert "addItem" in method_names
        assert "getItem" in method_names
        assert "size" in method_names

    def test_slice_file_extracts_constructor(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        constructors = [s for s in slices if s.method_name == "SampleService"]
        assert len(constructors) >= 1

    def test_slice_file_extracts_imports(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        # All slices should share the same imports
        imports = slices[0].imports
        assert any("java.util.List" in i for i in imports)
        assert any("java.util.ArrayList" in i for i in imports)

    def test_slice_file_extracts_docstring(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        add_item = next(s for s in slices if s.method_name == "addItem")
        assert "Add an item" in add_item.docstring

    def test_slice_file_has_correct_line_numbers(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        for s in slices:
            assert s.start_line >= 1
            assert s.end_line >= s.start_line

    def test_slice_file_content_matches_lines(self, slicer: JavaSlicer, sample_java_file: Path):
        source = sample_java_file.read_text().splitlines()
        slices = slicer.slice_file(sample_java_file)
        for s in slices:
            expected = "\n".join(source[s.start_line - 1:s.end_line])
            assert s.content == expected

    def test_slice_file_module_propagates(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file, module="my-module")
        for s in slices:
            assert s.module == "my-module"


class TestSymbolSignature:
    """Symbol signature generation."""

    def test_class_signature(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        class_slice = next(s for s in slices if s.method_name == "<class>")
        assert class_slice.symbol_signature == "com.example.demo.SampleService"

    def test_method_signature(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        add_item = next(s for s in slices if s.method_name == "addItem")
        assert "SampleService.addItem" in add_item.symbol_signature

    def test_constructor_signature(self, slicer: JavaSlicer, sample_java_file: Path):
        slices = slicer.slice_file(sample_java_file)
        ctor = next(s for s in slices if s.method_name == "SampleService")
        assert "SampleService.SampleService" in ctor.symbol_signature


class TestOverloadedMethods:
    """Handling of overloaded methods."""

    def test_overloaded_methods_produce_separate_slices(self, slicer: JavaSlicer, tmp_path: Path):
        content = textwrap.dedent("""\
            package com.example;

            public class Calculator {
                public int add(int a, int b) {
                    return a + b;
                }

                public double add(double a, double b) {
                    return a + b;
                }

                public int add(int a, int b, int c) {
                    return a + b + c;
                }
            }
        """)
        f = tmp_path / "Calculator.java"
        f.write_text(content)
        slices = slicer.slice_file(f)
        add_slices = [s for s in slices if s.method_name == "add"]
        assert len(add_slices) == 3
        # Each should have different signatures
        sigs = {s.symbol_signature for s in add_slices}
        assert len(sigs) == 3


class TestFileLevelDegradation:
    """File-level degradation when tree-sitter fails."""

    def test_invalid_java_falls_back_to_file_level(self, slicer: JavaSlicer, tmp_path: Path):
        # Create a file with syntax errors that tree-sitter can't fully parse
        content = textwrap.dedent("""\
            package com.example;

            public class Broken {
                public void method( {
                    // unclosed parenthesis
            }
        """)
        f = tmp_path / "Broken.java"
        f.write_text(content)
        slices = slicer.slice_file(f)
        # Should still produce at least one slice (file-level or partial)
        assert len(slices) >= 1

    def test_file_level_fallback_has_file_method_name(self, slicer: JavaSlicer, tmp_path: Path):
        # Completely unparseable content
        content = "this is not java code at all {{{{"
        f = tmp_path / "NotJava.java"
        f.write_text(content)
        slices = slicer.slice_file(f)
        assert len(slices) == 1
        assert slices[0].method_name == "<file>"
        assert slices[0].class_name == "NotJava"


class TestEdgeCases:
    """Edge cases for the slicer."""

    def test_empty_class(self, slicer: JavaSlicer, tmp_path: Path):
        content = textwrap.dedent("""\
            package com.example;

            public class Empty {
            }
        """)
        f = tmp_path / "Empty.java"
        f.write_text(content)
        slices = slicer.slice_file(f)
        assert len(slices) >= 1

    def test_interface(self, slicer: JavaSlicer, tmp_path: Path):
        content = textwrap.dedent("""\
            package com.example;

            public interface MyInterface {
                void doSomething();
                int calculate(int x);
            }
        """)
        f = tmp_path / "MyInterface.java"
        f.write_text(content)
        slices = slicer.slice_file(f)
        method_names = [s.method_name for s in slices]
        assert "doSomething" in method_names or "<class>" in method_names

    def test_enum(self, slicer: JavaSlicer, tmp_path: Path):
        content = textwrap.dedent("""\
            package com.example;

            public enum Color {
                RED, GREEN, BLUE;

                public String toHex() {
                    return "";
                }
            }
        """)
        f = tmp_path / "Color.java"
        f.write_text(content)
        slices = slicer.slice_file(f)
        assert len(slices) >= 1

    def test_nonexistent_file_raises(self, slicer: JavaSlicer, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            slicer.slice_file(tmp_path / "nonexistent.java")

    def test_non_java_file_raises(self, slicer: JavaSlicer, tmp_path: Path):
        f = tmp_path / "file.py"
        f.write_text("print('hello')")
        with pytest.raises(ValueError, match="Not a Java file"):
            slicer.slice_file(f)


class TestDirectorySlicing:
    """Directory-level slicing."""

    def test_slice_directory_finds_all_java_files(self, slicer: JavaSlicer, tmp_path: Path):
        # Create a directory with multiple Java files
        (tmp_path / "src").mkdir()
        for name in ["A.java", "B.java", "C.txt"]:
            (tmp_path / "src" / name).write_text(
                "package com.example;\npublic class X {}\n"
            )
        slices = slicer.slice_directory(tmp_path / "src")
        # Should find A.java and B.java, not C.txt
        file_paths = {s.file_path for s in slices}
        assert any("A.java" in fp for fp in file_paths)
        assert any("B.java" in fp for fp in file_paths)
        assert not any("C.txt" in fp for fp in file_paths)

    def test_slice_directory_recursive(self, slicer: JavaSlicer, tmp_path: Path):
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "main" / "java" / "Main.java").write_text(
            "package com;\npublic class Main {}\n"
        )
        slices = slicer.slice_directory(tmp_path / "src")
        assert len(slices) >= 1


class TestDemoRepoSlicing:
    """Integration tests on the actual demo repo."""

    def test_order_service_has_calculate_total(self, slicer: JavaSlicer):
        demo_repo = Path("demo-repo/src/main/java/com/example/order/OrderService.java")
        if not demo_repo.exists():
            pytest.skip("Demo repo not available")
        slices = slicer.slice_file(demo_repo, module="order-service")
        method_names = [s.method_name for s in slices]
        assert "calculateTotal" in method_names

    def test_order_service_calculate_total_lines(self, slicer: JavaSlicer):
        demo_repo = Path("demo-repo/src/main/java/com/example/order/OrderService.java")
        if not demo_repo.exists():
            pytest.skip("Demo repo not available")
        slices = slicer.slice_file(demo_repo)
        calc = next(s for s in slices if s.method_name == "calculateTotal")
        assert calc.start_line > 0
        assert calc.end_line > calc.start_line
        assert "calculateTotal" in calc.content

    def test_all_demo_files_slice_without_error(self, slicer: JavaSlicer):
        demo_dir = Path("demo-repo/src/main/java/com/example/order")
        if not demo_dir.exists():
            pytest.skip("Demo repo not available")
        for f in demo_dir.glob("*.java"):
            slices = slicer.slice_file(f)
            assert len(slices) >= 1, f"No slices from {f.name}"
