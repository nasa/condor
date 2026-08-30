def options_to_kwargs(new_cls, attr_name="Options"):
    """Process a model class and create the kwarg dictionary for the :class:`Options`"""
    opts = getattr(new_cls, attr_name, None)
    if opts is not None:
        backend_option = {
            k: v for k, v in opts.__dict__.items() if not k.startswith("_")
        }
    else:
        backend_option = {}
    return backend_option
