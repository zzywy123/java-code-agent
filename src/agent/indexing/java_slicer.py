"""Java code method-level slicer using tree-sitter.

Parses Java source files into method-level slices with metadata:
module, package, class_name, method_name, file_path, start_line, end_line,
content, imports, docstring, symbol_signature.

When tree-sitter fails to parse a file, falls back to file-level degradation:
the entire file becomes a single chunk with method_name='<file>'.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent.models import CodeSlice

# tree-sitter imports — fail loudly if unavailable
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

_JAVA_LANGUAGE = Language(tsjava.language())


def _create_parser() -> Parser:
    """Create a tree-sitter parser for Java."""
    return Parser(_JAVA_LANGUAGE)


def _node_text(node: Any, source_bytes: bytes) -> str:
    """Extract text content of a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_docstring(node: Any, source_bytes: bytes) -> str:
    """Extract the docstring/comment preceding a node.

    Looks for a preceding comment or javadoc block.
    """
    # Check if there's a preceding sibling that is a comment
    parent = node.parent
    if parent is None:
        return ""

    child_count = parent.child_count
    node_index = None
    for i in range(child_count):
        if parent.child(i).id == node.id:
            node_index = i
            break

    if node_index is None or node_index == 0:
        return ""

    # Walk backwards to find comment
    comments = []
    for i in range(node_index - 1, -1, -1):
        sibling = parent.child(i)
        if sibling.type in ("block_comment", "line_comment", "comment"):
            comments.insert(0, _node_text(sibling, source_bytes))
        elif sibling.type == "\n" or sibling.type == "\r\n":
            continue
        else:
            break

    return "\n".join(comments) if comments else ""


def _get_parameter_types(node: Any, source_bytes: bytes) -> list[str]:
    """Extract formal parameter types from a method_declaration node."""
    types = []
    for child in node.children:
        if child.type == "formal_parameters":
            for param_child in child.children:
                if param_child.type == "formal_parameter" or param_child.type == "spread_parameter":
                    # First child of formal_parameter is the type
                    type_node = param_child.child(0)
                    if type_node:
                        types.append(_node_text(type_node, source_bytes))
    return types


def _get_class_name(node: Any, source_bytes: bytes) -> str:
    """Extract class/enum/interface name from a declaration node."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source_bytes)
    return "<anonymous>"


def _get_method_name(node: Any, source_bytes: bytes) -> str:
    """Extract method name from a method_declaration node."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source_bytes)
    return "<init>"  # constructor


def _get_package(node: Any, source_bytes: bytes) -> str:
    """Extract package name from a program node."""
    for child in node.children:
        if child.type == "package_declaration":
            # Find the scoped_identifier child
            for pkg_child in child.children:
                if pkg_child.type in ("scoped_identifier", "identifier"):
                    return _node_text(pkg_child, source_bytes)
    return ""


def _get_imports(node: Any, source_bytes: bytes) -> list[str]:
    """Extract import statements from a program node."""
    imports = []
    for child in node.children:
        if child.type == "import_declaration":
            imports.append(_node_text(child, source_bytes))
    return imports


class JavaSlicer:
    """Slices Java source files into method-level code chunks.

    Uses tree-sitter for AST parsing. When parsing fails, falls back
    to file-level degradation (entire file as one chunk).
    """

    def slice_file(self, file_path: Path, module: str = "") -> list[CodeSlice]:
        """Slice a single Java file into method-level chunks.

        Args:
            file_path: Absolute path to the Java file
            module: Module name (from pom.xml artifactId)

        Returns:
            List of CodeSlice objects

        Raises:
            RuntimeError: If tree-sitter fails AND file-level degradation also fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.suffix == ".java":
            raise ValueError(f"Not a Java file: {file_path}")

        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise RuntimeError(f"Cannot read file (not UTF-8): {file_path}")

        source_bytes = source.encode("utf-8")

        # Try tree-sitter parsing
        try:
            parser = _create_parser()
            tree = parser.parse(source_bytes)
            root = tree.root_node

            # Check for parse errors
            if root.has_error:
                return self._file_level_fallback(file_path, source, module)

            return self._extract_slices(root, source_bytes, source, file_path, module)

        except Exception:
            # tree-sitter failure — file-level degradation
            return self._file_level_fallback(file_path, source, module)

    def slice_directory(self, dir_path: Path, module: str = "") -> list[CodeSlice]:
        """Slice all Java files in a directory tree.

        Args:
            dir_path: Root directory to scan
            module: Module name

        Returns:
            List of CodeSlice objects from all Java files
        """
        all_slices: list[CodeSlice] = []
        for java_file in sorted(dir_path.rglob("*.java")):
            # Skip test files (optional — include them for full indexing)
            try:
                slices = self.slice_file(java_file, module)
                all_slices.extend(slices)
            except Exception:
                # Skip files that fail completely
                continue
        return all_slices

    def _extract_slices(
        self,
        root: Any,
        source_bytes: bytes,
        source: str,
        file_path: Path,
        module: str,
    ) -> list[CodeSlice]:
        """Extract method-level slices from a parsed AST."""
        package = _get_package(root, source_bytes)
        imports = _get_imports(root, source_bytes)
        slices: list[CodeSlice] = []

        source_lines = source.splitlines()

        # Walk the AST for class/enum/interface declarations
        self._walk_declarations(
            root, source_bytes, source_lines, file_path,
            package, imports, module, slices,
            enclosing_class="",
        )

        # If no slices found (e.g. pure interface with no methods),
        # create a file-level slice
        if not slices:
            slices.append(self._make_file_slice(
                source, source_lines, file_path, package, imports, module,
            ))

        return slices

    def _walk_declarations(
        self,
        node: Any,
        source_bytes: bytes,
        source_lines: list[str],
        file_path: Path,
        package: str,
        imports: list[str],
        module: str,
        slices: list[CodeSlice],
        enclosing_class: str,
    ) -> None:
        """Recursively walk AST to find class and method declarations."""
        for child in node.children:
            if child.type in ("class_declaration", "enum_declaration", "interface_declaration", "record_declaration"):
                class_name = _get_class_name(child, source_bytes)
                full_class = f"{enclosing_class}.{class_name}" if enclosing_class else class_name
                docstring = _get_docstring(child, source_bytes)

                # Add a class-level slice
                start_line = child.start_point[0] + 1  # 1-based
                end_line = child.end_point[0] + 1
                content = "\n".join(source_lines[start_line - 1:end_line])

                slices.append(CodeSlice(
                    module=module,
                    package=package,
                    class_name=full_class,
                    method_name="<class>",
                    file_path=file_path.as_posix(),
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    imports=imports,
                    docstring=docstring,
                    symbol_signature=f"{package}.{full_class}" if package else full_class,
                ))

                # Now find methods inside this class
                self._walk_methods(
                    child, source_bytes, source_lines, file_path,
                    package, imports, module, full_class, slices,
                )

                # Recurse into nested classes
                self._walk_declarations(
                    child, source_bytes, source_lines, file_path,
                    package, imports, module, slices, full_class,
                )

    def _walk_methods(
        self,
        class_node: Any,
        source_bytes: bytes,
        source_lines: list[str],
        file_path: Path,
        package: str,
        imports: list[str],
        module: str,
        class_name: str,
        slices: list[CodeSlice],
    ) -> None:
        """Find method declarations inside a class/enum/interface node."""
        # Walk into class_body / enum_body / interface_body
        for child in class_node.children:
            if child.type in ("class_body", "enum_body", "interface_body", "record_body"):
                for member in child.children:
                    if member.type in (
                        "method_declaration",
                        "constructor_declaration",
                    ):
                        method_name = _get_method_name(member, source_bytes)
                        param_types = _get_parameter_types(member, source_bytes)
                        docstring = _get_docstring(member, source_bytes)

                        start_line = member.start_point[0] + 1
                        end_line = member.end_point[0] + 1
                        content = "\n".join(source_lines[start_line - 1:end_line])

                        # Build symbol signature: package.ClassName.methodName(ParamType1,ParamType2)
                        params_str = ",".join(param_types)
                        sig = f"{package}.{class_name}.{method_name}({params_str})" if package else f"{class_name}.{method_name}({params_str})"

                        slices.append(CodeSlice(
                            module=module,
                            package=package,
                            class_name=class_name,
                            method_name=method_name,
                            file_path=file_path.as_posix(),
                            start_line=start_line,
                            end_line=end_line,
                            content=content,
                            imports=imports,
                            docstring=docstring,
                            symbol_signature=sig,
                        ))

    def _make_file_level_fallback(
        self,
        file_path: Path,
        source: str,
        module: str,
    ) -> list[CodeSlice]:
        """File-level degradation when tree-sitter fails."""
        return self._file_level_fallback(file_path, source, module)

    def _file_level_fallback(
        self,
        file_path: Path,
        source: str,
        module: str,
    ) -> list[CodeSlice]:
        """Create a single file-level chunk when tree-sitter parsing fails.

        method_name is set to '<file>' to indicate degradation.
        """
        source_lines = source.splitlines()

        # Try to extract package from text
        package = ""
        imports: list[str] = []
        for line in source_lines:
            stripped = line.strip()
            if stripped.startswith("package "):
                package = stripped.replace("package ", "").rstrip(";").strip()
            elif stripped.startswith("import "):
                imports.append(stripped.rstrip(";"))

        # Try to extract class name from filename
        class_name = file_path.stem  # e.g. "OrderService"

        return [CodeSlice(
            module=module,
            package=package,
            class_name=class_name,
            method_name="<file>",
            file_path=file_path.as_posix(),
            start_line=1,
            end_line=len(source_lines),
            content=source,
            imports=imports,
            docstring="",
            symbol_signature=f"{package}.{class_name}" if package else class_name,
        )]
