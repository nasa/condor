from condor import backend
from condor.fields import (
    BaseElement,
)


class ElementMap(dict):
    """dictionary for mapping elements (typically to other backend expressions), with
    conveniences for getting or keying off a specific element attribute"""

    def as_(self, left_attr):
        """return a dictionary where the key is an attribute of the element"""
        # TODO: turn this into a yield statement?
        return {
            (
                getattr(k, left_attr)
                if isinstance(k, BaseElement)
                else k
                if left_attr == "backend_repr" and isinstance(k, backend.symbol_class)
                else None
            ): v
            for k, v in self.items()
        }

    def get(self, *args, **kwargs):
        """Get a list of field elements where every field matches kwargs

        If only one, return element without list wrapper.
        """
        # TODO: what's the lightest-weight way to be able query? these should get called
        # very few times, so hopefully don't need to stress too much about
        # implementation
        # would be nice to be able to follow references, etc. can match be used?
        if len(args) == 1 and not kwargs:
            field_value = args[0]
            if isinstance(field_value, backend.symbol_class):
                kwargs.update(backend_repr=field_value)

        items = []
        for item in self:
            this_item = True
            for field_name, field_value in kwargs.items():
                item_value = getattr(item, field_name)
                if isinstance(item_value, backend.symbol_class):
                    if not isinstance(field_value, backend.symbol_class):
                        this_item = False
                        break
                    this_item = this_item and backend.symbol_is(item_value, field_value)
                elif isinstance(item_value, BaseElement):
                    if item_value.__class__ is not field_value.__class__:
                        this_item = False
                        break
                    this_item = this_item and item_value is field_value
                else:
                    this_item = this_item and item_value == field_value
                if not this_item:
                    break
            if this_item:
                items.append(item)
        if len(items) == 1:
            return items[0]
        return items
