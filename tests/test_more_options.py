import condor as co
from condor.implementations.utils import options_to_kwargs


def test_options_to_kwargs():
    class ExampleModel(co.ExplicitSystem):
        class Options:
            in_options = True

        class MiscOptions:
            in_misc_options = True

    assert options_to_kwargs(ExampleModel)["in_options"]
    assert options_to_kwargs(ExampleModel, attr_name="MiscOptions")["in_misc_options"]
