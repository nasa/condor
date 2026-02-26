import ast
from typing import Any

import sphinx
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter, Documenter, ObjectMember, bool_option
from sphinx.pycode.parser import (
    AfterCommentParser,
    VariableCommentPicker,
    comment_re,
    dedent_docstring,
    get_assign_targets,
    indent_re,
)
from sphinx.util.typing import ExtensionMetadata

from condor.models import Model, SubmodelTemplateType

sphinx_major_version = int(sphinx.__version__.split(".")[0])

# this will make it behave like sphinx<9.0,
# autodoc_use_legacy_class_basedautodoc_use_legacy_class_based = True


class CondorModelCommentPicker(VariableCommentPicker):
    def __init__(self, buffers: list[str], encoding: str, sys: Model) -> None:
        super().__init__(buffers, encoding)
        self._sys = sys

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        """Handles Assign node; copy/paste/modify from VariableCommentPicker"""
        targets = get_assign_targets(node)
        assigned_field_names = [
            f._name
            for f in self._sys._meta.assigned_fields
            if f._add_to_namespace or f._add_to_namespace_override
        ]
        varnames: list[str] = []
        is_assigned_field = False
        for target in targets:
            if isinstance(target, ast.Name) and len(targets) == 1:
                return super().visit_Assign(node)
            # TODO: fields should get hooks to parse variable name / decide
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in assigned_field_names
            ):
                is_assigned_field = True
                varnames.append(target.attr)
            elif isinstance(target, ast.Subscript):
                pass

        current_line = self.get_line(node.lineno)
        if is_assigned_field:
            pass

        # remainder same as super()
        if hasattr(node, "annotation") and node.annotation:
            for varname in varnames:
                self.add_variable_annotation(varname, node.annotation)
        elif hasattr(node, "type_comment") and node.type_comment:
            for varname in varnames:
                self.add_variable_annotation(varname, node.type_comment)  # type: ignore[arg-type]

        # check comments after assignment
        parser = AfterCommentParser(
            [current_line[node.col_offset :]] + self.buffers[node.lineno :]
        )
        parser.parse()
        if parser.comment and comment_re.match(parser.comment):
            for varname in varnames:
                self.add_variable_comment(
                    varname, comment_re.sub("\\1", parser.comment)
                )
                self.add_entry(varname)
            return

        # check comments before assignment
        if indent_re.match(current_line[: node.col_offset]):
            comment_lines = []
            for i in range(node.lineno - 1):
                before_line = self.get_line(node.lineno - 1 - i)
                if comment_re.match(before_line):
                    comment_lines.append(comment_re.sub("\\1", before_line))
                else:
                    break

            if comment_lines:
                comment = dedent_docstring("\n".join(reversed(comment_lines)))
                for varname in varnames:
                    self.add_variable_comment(varname, comment)
                    self.add_entry(varname)
                return

        # not commented (record deforders only)
        for varname in varnames:
            self.add_entry(varname)


class CondorModelDocumenter(ClassDocumenter):
    objtype = "model"
    directivetype = ClassDocumenter.objtype
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)
    option_spec["hex"] = bool_option

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Documenter
    ) -> bool:
        try:
            return_val = issubclass(member, Model)
            if return_val and member._meta.model_name == "RangeRdotSensor":
                print(member)
            return return_val
        except TypeError:
            return False

    def add_content(
        self,
        more_content: StringList | None,
    ) -> None:
        if self.analyzer:
            # reproduce part of parse chain: sphinx.pycode.ModuleAnalyzer.analyze() -->
            # pycode.Parser.parse() --> pycode.Parser.parse_comments() -->
            # pycode.VariableCOmmentPicker.visit()
            code = self.analyzer.code
            tree = ast.parse(code, type_comments=True)
            picker = CondorModelCommentPicker(
                code.splitlines(True),
                "utf-8",
                self.object,
            )
            picker.visit(tree)
            attr_docs = {}
            for scope, comment in picker.comments.items():
                if comment:
                    attr_docs[scope] = [*comment.splitlines(), ""]
                else:
                    attr_docs[scope] = [""]
            self.analyzer.attr_docs = attr_docs
            self.analyzer.tagorder = picker.deforders
        super().add_content(more_content)

    def get_object_members(self, want_all: bool) -> tuple[bool, list[ObjectMember]]:
        members_check_module, members = super().get_object_members(want_all)
        # original_members = members
        members = [m for m in members if not isinstance(m.object, SubmodelTemplateType)]
        members = [
            m
            for m in members
            if m.object is not self.object._meta.inherited_methods.get(m.__name__, None)
        ]
        return members_check_module, members


def setup(app: Sphinx) -> ExtensionMetadata:
    # autodoc_use_legacy_class_basedautodoc_use_legacy_class_based
    app.setup_extension("sphinx.ext.autodoc")  # Require autodoc extension
    app.add_autodocumenter(CondorModelDocumenter)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
    }
