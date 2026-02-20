import condor


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
