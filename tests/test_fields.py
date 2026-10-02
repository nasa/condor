import casadi
import numpy as np
import pytest

import condor
from condor.backend import operators


def test_create_from():
    class Sys1(condor.ExplicitSystem):
        a = input()
        b = input()

        output.c = a + b

    class Sys2(condor.ExplicitSystem):
        a_lias = input()

        sys1_in = input.create_from(Sys1.input, a=a_lias)
        sys1_out = Sys1(**sys1_in)

        output.d = sys1_out.c + 1

    out = Sys2(a_lias=10, b=2)
    assert out.d == 10 + 2 + 1


def test_create_from_different_field_types():
    class Sys1(condor.ExplicitSystem):
        a = input()
        b = input()

        output.c = a + b

    class Sys2(condor.AlgebraicSystem):
        a = variable()
        c_target = parameter()

        sys1_in = parameter.create_from(Sys1.input, a=a)
        sys1_out = Sys1(**sys1_in)

        residual(sys1_out.c == c_target)

    out = Sys2(b=3, c_target=10)
    assert out.a == 7


def test_dict_unpack():
    class Sys(condor.ExplicitSystem):
        a = input()
        b = input()

        output.c = a + b

    assert dict(**Sys.input) == {"a": Sys.a.backend_repr, "b": Sys.b.backend_repr}
    assert dict(**Sys.output) == {"c": Sys.c.backend_repr}

    sys = Sys(a=1, b=-2)
    assert dict(**sys.input) == {"a": 1, "b": -2}
    assert dict(**sys.output) == {"c": -1}


@pytest.fixture
def value_system():
    class ValueSystem(condor.ExplicitSystem):
        x = input()
        vector = input(shape=(2, 1))
        matrix = input(shape=(2, 2))

        output.result = x + vector[0] + matrix[0, 0]

    return ValueSystem


def test_field_values_dict_of_symbols(value_system):
    instance = value_system(x=2, vector=np.array([3, 5]), matrix=np.eye(2))
    values = instance.input.dict_of()

    assert list(values) == value_system.input.list_of("backend_repr")
    for element in value_system.input:
        assert values[element.backend_repr] is getattr(instance.input, element.name)
    assert len(values) == 3


def test_field_values_dict_of_names(value_system):
    instance = value_system(x=2, vector=np.array([3, 5]), matrix=np.eye(2))
    values = instance.input.dict_of("name")

    assert list(values) == ["x", "vector", "matrix"]
    assert values["x"] == instance.input.x
    assert values["vector"] is instance.input.vector
    assert values["matrix"] is instance.input.matrix
    assert values["vector"].shape == instance.input.vector.shape
    assert values["matrix"].shape == (2, 2)
    # The existing conversion keeps its independent deep-copy semantics.
    copied = instance.input.asdict()
    assert copied["matrix"] is not instance.input.matrix
    np.testing.assert_array_equal(copied["matrix"], values["matrix"])


def test_field_values_dict_of_symbolic_values(value_system):
    symbolic = value_system.input.dataclass_of()
    values = symbolic.dict_of()

    for element in value_system.input:
        assert values[element.backend_repr] is element.backend_repr


def test_field_values_dict_of_substitution(value_system):
    instance = value_system(x=2, vector=np.array([3, 5]), matrix=np.eye(2))
    expression = value_system.result.backend_repr
    evaluated = operators.substitute(expression, instance.input.dict_of())

    assert float(casadi.evalf(evaluated)) == pytest.approx(instance.result.item())


def test_field_values_dict_of_output(value_system):
    instance = value_system(x=2, vector=np.array([3, 5]), matrix=np.eye(2))
    result = instance.output.dict_of()

    assert len(result) == 1
    assert result[value_system.result.backend_repr] is instance.output.result


def test_field_values_dict_of_empty_field():
    class Constant(condor.ExplicitSystem):
        output.value = 1

    instance = Constant()
    assert instance.input.dict_of() == {}
    assert instance.input.dict_of("name") == {}


def test_field_values_dict_of_invalid_attribute(value_system):
    values = value_system.input.dataclass_of()
    with pytest.raises(AttributeError, match="no_such_attribute"):
        values.dict_of("no_such_attribute")


def test_field_values_dict_of_returns_new_mapping(value_system):
    values = value_system.input.dataclass_of()
    first = values.dict_of()
    second = values.dict_of()
    first.clear()

    assert len(second) == 3
    assert len(values.dict_of()) == 3
