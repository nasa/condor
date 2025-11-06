import dataclasses as dc
import logging

from condor import backend, implementations

# TODO: figure out how to make this an option/setting like django?
# from condor.backends import default as backend
from condor.fields import (
    AssignedField,
    BaseElement,
    Direction,
    Field,
    FreeField,
    MatchedField,
    WithDefaultField,
)

log = logging.getLogger(__name__)


@dc.dataclass
class BaseModelMetaData:
    """Holds metadata for models"""

    model_name: str = ""
    independent_fields: list = dc.field(default_factory=list)
    matched_fields: list = dc.field(default_factory=list)
    assigned_fields: list = dc.field(default_factory=list)

    input_fields: list = dc.field(default_factory=list)
    output_fields: list = dc.field(default_factory=list)
    internal_fields: list = dc.field(default_factory=list)

    input_names: list = dc.field(default_factory=list)
    output_names: list = dc.field(default_factory=list)

    submodels: list = dc.field(default_factory=list)
    embedded_models: dict = dc.field(default_factory=dict)
    bind_embedded_models: bool = True

    inherited_items: dict = dc.field(default_factory=dict)
    template: object = None

    user_set: dict = dc.field(default_factory=dict)
    backend_repr_elements: dict = dc.field(default_factory=backend.SymbolCompatibleDict)
    options: object = None

    # assembly/component can also get children/parent
    # assembly/components get inheritance rules? yes, submodels don't need it -- only
    # attach to primary. or should events be assemblies to re-use them? probably not --
    # re-use other things and embed but update itself is ODE system specific

    # trajectory analysis gets an "exclude_modes" and "exclude_events" list?
    # no -- the metaclass gets kwarg and copies the events that will be kept

    @property
    def all_fields(self):
        return self.input_fields + self.output_fields + self.internal_fields

    @property
    def dependent_fields(self):
        return [f for f in self.all_fields if f not in self.independent_fields]

    @property
    def noninput_fields(self):
        return self.output_fields + self.internal_fields

    def copy_update(self, **new_meta_kwargs):
        meta_kwargs = {}
        for field in dc.fields(self):
            field_val = getattr(self, field.name)
            if isinstance(field_val, list):
                # shallow copy
                field_val = [item for item in field_val]
            meta_kwargs[field.name] = field_val
        meta_kwargs.update(new_meta_kwargs)
        new_meta = self.__class__(**meta_kwargs)
        return new_meta


class AnnotationsDict(dict):
    def __init__(self, *args, cls_dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_dict = cls_dict

    def __setitem__(self, attr_name, attr_val):
        if isinstance(attr_val, AssignedField):
            setattr(attr_val, attr_name, self.cls_dict[attr_name])

        super().__setitem__(attr_name, attr_val)


# appears in __new__ as attrs
class BaseCondorClassDict(dict):
    def __init__(
        self,
        *args,
        model_name="",
        # copy_fields=[], primary=None,
        meta=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.from_outer = {}
        self.meta = meta
        self.kwargs = kwargs
        self.args = args
        self.user_setting = False

    def set_outer(self, **from_outer):
        self.from_outer = from_outer
        self.meta.independent_fields.extend(
            [
                v
                for k, v in from_outer.items()
                if isinstance(v, FreeField)  # and k not in self.copy_fields
            ]
        )

    def __getitem__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)

    def __setitem__(self, attr_name, attr_val):
        if attr_name == "__annotations__" and len(attr_val) == 0:
            super().__setitem__(attr_name, AnnotationsDict(cls_dict=self))
            return

        if (
            self.meta.template is not None
            and attr_name in type(self.meta.template).reserved_words
            and self.user_setting
        ):
            msg = (
                f"Attempting to set {attr_name}={attr_val}, but {attr_name} is a "
                "reserved word"
            )
            raise ValueError(msg)

        log.debug("setting %s to %s on %s", attr_name, attr_val, self.meta.model_name)

        if isinstance(attr_val, FreeField):
            self.meta.independent_fields.append(attr_val)
        if isinstance(attr_val, MatchedField):
            self.meta.matched_fields.append(attr_val)
        if isinstance(attr_val, AssignedField):
            self.meta.assigned_fields.append(attr_val)
        if isinstance(attr_val, Field):
            if attr_val._direction == Direction.input:
                self.meta.input_fields.append(attr_val)
            if attr_val._direction == Direction.output:
                self.meta.output_fields.append(attr_val)
            if attr_val._direction == Direction.internal:
                self.meta.internal_fields.append(attr_val)

            attr_val.prepare(self)

        if isinstance(attr_val.__class__, BaseModelType):
            # if all input symbols are from this model, embed
            # otherwise it's a stray reference
            can_embed = True
            for input_val in attr_val.input_kwargs.values():
                if not can_embed:
                    break
                if isinstance(input_val, backend.symbol_class):
                    for input_sym in backend.symbols_in(input_val):
                        if input_sym not in self.meta.backend_repr_elements:
                            can_embed = False
                            break
            if getattr(attr_val, "name", ""):
                if can_embed:
                    self.meta.embedded_models[attr_val.name] = attr_val
                super().__setitem__(attr_val.name, attr_val)
            else:
                if can_embed:
                    self.meta.embedded_models[attr_name] = attr_val
                attr_val.name = attr_name
        if isinstance(attr_val, BaseElement) and not attr_val.name:
            attr_val.name = attr_name
        if isinstance(attr_val, backend.symbol_class):
            # from a FreeField
            element = self.meta.backend_repr_elements.get(attr_val, None)
            if element is not None:
                if element.name:
                    pass
                elif attr_name:
                    element.name = attr_name
                else:
                    element.name = (
                        f"{field._model_name}_{field._name}"
                        f"_{field._elements.index(element)}"
                    )

            # TODO: MatchedField in "free" mode
            # TODO: from the output of a subsystem? Does this case matter?

            else:
                pass
                # DONT pass these on. they don't work to use in the
                # another model since they don't get bound correctly, then just have
                # dangling casadi expression/symbol. Event functions can just use
                # ODESystem.t, or maybe handle as a special case? Could  check to
                # see if it's in the template, and only keep those?

        # TODO: other possible attr types to process:
        # maybe field dataclass, if we care about intervening in eventual assignment
        # deferred subsystems, but perhaps this will never get hit? need to figure
        # out how a model with deferred subsystem behaves

        if self.user_setting:
            self.meta.user_set[attr_name] = attr_val

        out = super().__setitem__(attr_name, attr_val)

        if (
            self.user_setting and hasattr(attr_val, "update_name")  # implies element?
        ):
            if hasattr(attr_val, "field_type") and (
                attr_val.field_type._cls_dict is not self
            ):
                return out

            attr_val.update_name(attr_name)
        return out


class DynamicLink:
    def __init__(self, cls_dict):
        self.cls_dict = cls_dict

    def __call__(self, **kwargs):
        """
        replace the string values of a dictionary with the actual object with that
        address

        dynamic_link(xx='state.x')

        returns a dictionary where the key 'xx' points to the state with name 'x'

        maybe should be in ModelClassDict (for ModelType specifically) since Templates
        shouldn't need this. Then ModelType can be responsible for replacing it
        """
        for k, v in kwargs.items():
            path_list = v.split(".")
            use_val = self.cls_dict.__getitem__(path_list[0])
            for path_step in path_list[1:]:
                use_val = getattr(use_val, path_step)
            if isinstance(use_val, FreeField):
                use_val = use_val(name=k)
            kwargs[k] = use_val

        return kwargs


# TODO: a way to update options -- options should become attributes on implementation
# that are used on every call
class Options:
    """
    Django uses a class that owns `class Meta` attribute from model, and then processes
    it. Has `abstract` attribute to not build table -- is that what an extended template
    is?


    re-writing implementations to be backend agnostic (through "wrapper" layer ), the
    option class won't need to be named after a backend; maybe just take __solver__
    attribute to map the solver funciton [it could actually be the callable?
    scipy.minimize, lol sorry Kenny, you can change it after release once I'm bored]
    implementations dictionary needs to key on solver and be the arg/kwarg location for
    the numeric callbacks (for this model/paramters) and a few other things like initial
    state. Current version is pretty close, hopefully

    Goal: ensure API works to pass in *args and **kwargs easily; Do later
    probably need a __args attr and then rest as kwargs. Common pattern of reserving
    wordds in user code. Also applies to IO name check, assigned field

    to support changing solver options within an ipython session, maybe implementation
    actually has a create_solver and call_solver or something -- creating solver also
    caches any numeric callbacks

    All backend implementations must ship with reasonable defaults.

    Actually, don't need to inherit if it's backend agnostic -- can just be by name
    (which is what Django does)
    """

    pass
    # TODO: do meta programming so implementation enums are available without import?


class BaseModelType(type):
    """Metaclass for Condor model"""

    metadata_class = BaseModelMetaData
    dict_class = BaseCondorClassDict

    # not strictly necessary here since BaseModel doesn't need to exist but convenient
    # to have the logic to automatically fill this out on all subclasses of
    # BaseModelType
    baseclass_for_inheritance = None
    reserved_words = [
        "_meta",
        "reserved_words",
    ]

    def __init_subclass__(cls, **kwargs):
        cls.baseclass_for_inheritance = None
        # backreference for baseclass (e.g., ModelTemplate, Model, etc) not created
        # until

        # potential additional metadata:
        # these are only for templates? Or should templates get a custom metadata?

        # class dictionary subclass to over-write __set_item__ (to handle particular
        # attribute types/fields/etc, -- may be able to use callback on metclass or even
        # field

        # similar for meta attributes, will ultimately get used by __prepare__ or
        # __new__, I think?

    @classmethod
    def __prepare__(cls, name, bases, meta=None, **kwds):
        log.debug(
            "BaseModelType.__prepare__(cls=%s, name=%s, bases=%s, kwargs=%s)",
            cls,
            name,
            bases,
            kwds,
        )

        cls_dict = cls.prepare_create(name, bases, meta=meta, **kwds)
        if not cls_dict.meta.template:
            return cls_dict
        cls.prepare_populate(cls_dict, bases)
        cls_dict.user_setting = True
        return cls_dict

    @classmethod
    def prepare_create(cls, name, bases, meta=None, **kwds):
        """Create the class dictionary. Since the BaseModelType calls
        this, can only customize if the metaclass args on __prepare__ make their way to
        the metaclass

        """

        sup_dict = {}  # equivalent to super().__prepare__ unless I break something

        if meta is None:
            meta = cls.metadata_class(
                # primary=primary
                model_name=name,
            )

        # TODO: template should probably be first common Condor ancestor of all bases...
        # and/or kwd settable...
        for base in bases:
            if isinstance(base, BaseModelType):
                meta.template = base
                break

        cls_dict = cls.dict_class(
            model_name=name,
            meta=meta,
            **sup_dict,
        )

        # TODO: may need to search MRO resolution, not just bases, which without mixins
        # are just singletons. For fields and submodel classes, since each generation of
        # class is getting re-inherited, this is sufficient.

        return cls_dict

    def inherit_field(cls, cls_dict, new_direction=None):
        """Used to inherit a field from parent to new class dictionary (so it is
        available during class declaration)
        """
        v_init_kwargs = cls._init_kwargs.copy()
        for init_k, init_v in cls._init_kwargs.items():
            did_update = False
            if isinstance(init_v, (list, tuple)):
                init_v_iterable = init_v
                originaly_iterable = True
            else:
                init_v_iterable = [init_v]
                originaly_iterable = False

            use_kwarg = []
            for v in init_v_iterable:
                # TODO: is it OK to always assume cls_dict has the proper reference
                # injected
                if isinstance(v, Field) and v._name in cls_dict:
                    use_kwarg.append(cls_dict[v._name])
                    did_update = True
                    # v_init_kwargs[init_k] = cls_dict[v._name]
            if not originaly_iterable and use_kwarg:
                use_kwarg = use_kwarg[0]
            if did_update:
                v_init_kwargs[init_k] = use_kwarg
        inherited_field = cls.inherit(
            cls_dict.meta.model_name, field_type_name=cls._name, **v_init_kwargs
        )
        if new_direction is not None:
            inherited_field._direction = new_direction
        cls_dict[cls._name] = inherited_field
        cls_dict[cls._name].prepare(cls_dict)

    @classmethod
    def inherit_item(cls, cls_dict, field_from_inherited, meta, name, base, k, v):
        """inerit :attr:`k` = :attr:`v` as present in `base`'s locals dictionary
        field_from_inherited is a dictionary mapping base's fields to field on new class
        """
        if k in base._meta.inherited_items:
            log.debug("should this be injected? %s=%s", k, v)
            breakpoint()
        # inherit fields from base -- bound in __new__
        if isinstance(v, Field):
            if k in cls_dict:
                if cls_dict[k].__class__ is v.__class__:
                    pass  # compatible field inheritance
                else:
                    msg = f"inheriting incompatibility {base}.{k} = {v} to {name}"
                    raise ValueError(msg)
            else:
                cls.inherit_field(v, cls_dict)
            meta.inherited_items[k] = field_from_inherited[v] = cls_dict[v._name]

            if v._elements:
                for element in v:
                    new_elem = element.copy_to_field(
                        field_from_inherited[element.field_type]
                    )
                    if base.__dict__.get(element.name, None) is element:
                        cls_dict[new_elem.name] = new_elem.backend_repr

                log.debug(f"inheriting a non-empty field {k}={v} from {base} to {name}")
        # TODO: other possibilities to handle:
        # a BaseModelType would find a template/model attribute.
        # Submodels will get assigned like this, but Model should declare that
        # what about a Model(Template) that is being declared in the class boy?
        else:
            existing_attr = cls_dict.get(k, None)
            if isinstance(v, backend.symbol_class):
                elem_matching_backend_repr = meta.backend_repr_elements.get(v, None)
            else:
                elem_matching_backend_repr = None

            if existing_attr is not None or elem_matching_backend_repr is not None:
                if isinstance(existing_attr, BaseElement):
                    existing_attr = existing_attr.backend_repr
                elif isinstance(elem_matching_backend_repr, BaseElement):
                    existing_attr = elem_matching_backend_repr.backend_repr
                    # assume cls dict assignment??
                if (existing_attr is v) or (
                    isinstance(existing_attr, backend.symbol_class)
                    and backend.symbol_is(existing_attr, v)
                ):
                    # only ensure that this is marked as inherited, don't need
                    # to re-copy
                    if (inherited_attr := meta.inherited_items.get(k, None)) is None:
                        meta.inherited_items[k] = v
                    else:
                        if isinstance(inherited_attr, BaseElement):
                            inherited_attr = inherited_attr.backend_repr
                        if inherited_attr is not v:
                            msg = f"an inheritance bug, {base}.{k} = {v} to {name}"
                            raise ValueError(msg)

                    return
                elif cls.is_condor_attr(k, v):
                    msg = (
                        f"an inheritance incompatibility {base}.{k} = {v} does not "
                        f"match {name}.{k} = {cls_dict[k]}"
                    )
                    raise ValueError(msg)
            else:
                log.debug(
                    f"Independent element/symbol being inherited from {base}.{k} = {v} "
                    f"to {name}"
                )

            if isinstance(v, backend.symbol_class):
                original_v = v
                v = base._meta.backend_repr_elements.get(v, v)
                if isinstance(v, BaseElement) and v.field_type._name == "placeholder":
                    if (new_field := cls_dict.get("placeholder", None)) is not None:
                        # TODO: or just append element so it's the same backend repr?
                        new_v = v.copy_to_field(new_field)
                        v = new_v.backend_repr
                    else:
                        v = original_v

            if isinstance(v, BaseElement):
                # should only be used for placeholder and placeholder-adjacent
                if v.field_type in field_from_inherited:
                    new_elem = v.copy_to_field(field_from_inherited[v.field_type])
                    log.debug(
                        "Element %s=%s copied from %s to %s",
                        k,
                        v,
                        meta.template,
                        name,
                    )
                    cls_dict[k] = new_elem.backend_repr

                else:
                    log.debug(
                        "Element not inheriting %s=%s from %s to %s",
                        k,
                        v,
                        meta.template,
                        name,
                    )
                    cls_dict[k] = v
            elif isinstance(v, backend.symbol_class):
                # TODO: check what hits this. Previously, time dummy variable. But I
                # think that might be captured by placeholders now?
                log.debug(
                    "Element inheriting %s=%s from %s to %s",
                    k,
                    v,
                    meta.template,
                    name,
                )
                cls_dict[k] = v

        if k in cls_dict:
            meta.inherited_items[k] = cls_dict[k]

    @classmethod
    def prepare_populate(cls, cls_dict, bases):
        """Used to pre-populate the class namespace. Since the BaseModelType calls
        this, can only customize if the metaclass args on __prepare__ make their way to
        the metaclass
        """
        meta = cls_dict.meta
        name = meta.model_name
        template = meta.template

        # probably sufficient to do noninput_fields
        field_from_inherited = {}
        # field._inherits_from: field for field in meta.all_fields}

        # also need to iterate over bases/template, I guess?
        # cls.__mro__ might work. need to make sure the things in cls are fully copied,
        # as if user wrote it (again) -- equivalent to #include
        # but need to use same inheritance mechanisms. Actually, this is a problem when
        # the original/generic TrajectoryAnalysis is constructed, because the
        # placeholder field is on the ModelTemplate, not the SubmodelTemplate
        processed_mro = []
        for base in bases + template.__mro__[:-1]:
            # TODO verify that this is the right way to go. Go over the mro (excluding
            # object) and iterate over the user defined (not condor-machinery injected)
            # attributes to do a condor-based inheritance
            _dict = base._meta.user_set
            if base in processed_mro:
                continue
            processed_mro.extend(base.__mro__)

            for k, v in _dict.items():
                cls.inherit_item(cls_dict, field_from_inherited, meta, name, base, k, v)

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __repr__(cls):
        if cls.__bases__:
            return f"<{cls.__bases__[0].__name__}: {cls.__name__}>"
        else:
            return f"<{cls.__name__}>"

    @classmethod
    def creating_base_class_for_inheritance(cls, name):
        return cls.__name__.replace(name, "") == "Type" and (
            cls.baseclass_for_inheritance is None
        )

    @classmethod
    def is_condor_attr(cls, k, v):
        return (
            isinstance(v, (Field, BaseElement, backend.symbol_class))
            or k in cls.reserved_words
        )

    def __new__(cls, name, bases, attrs, **kwargs):
        """
        Processing of attrs in django...ModelBase is:
        pop meta
        iterate over attrs (old):
           separate out django-y attributes (_has_contribute_to_class(object) )
            from normal (`new_attrs`)

        - call super new with `new attrs`
        - start processing Meta options and setting up model accordingly, including...
            - contributing the contributable items. (only declared for this new class)
               <condor inheriting/initial binding>
            - a lot of table setup, linking etc. <implementation setup?>
            - go through all MRO, collect inheritable attributes and link for concrete
            class or deep copy and add to class (<condor inherit = copy, update, bind>)
        - new_cls._prepare (instance method on ModelBase = classmethod on Model?) does
          a few additional attributes, only one I understand is creating the manager
        - register model with app -- like creating back-reference to template's
          subclasses (instances)


        should follow pattern of separating out condor attributes
        -(isinstance(attr_val(Field, Element, backend_repr.__class__, ??)) as test)
        from non-condor attributes. Non-condor attributes get passed to __new__ early
        and then add appropriate condor attributes after-the fact. callback for
        processing attribute on Type, so only 1 for loop through dict?

        Then need to inherit from bases and/or template MRO



        """
        log.debug(
            "BaseModelType.__new__(mcs=%s, name=%s, bases=%s, attrs=[%s], %s)",
            cls,
            name,
            bases,
            attrs,
            kwargs,
        )
        attrs.user_setting = False
        # case 1: class Model -- provides machinery to make subsequent cases easier to
        # implement. "base model for inheritance"
        # case 2: template - library code that defines fields, etc that user code
        # inherits maybe call this model template?
        # from (case 3+). Implementations are tied to 2? "Library Models"?
        # case 3: User Model - inherits from template, defines the actual model that is
        # being analyzed, need to manipulate bases to swap for Model
        # case 4: Subclass of user model to extend it?
        # I don't think Submodel deviates from this, except perhaps disallowing
        # case 4
        # Generally, bases arg will be len <= 1. Just from the previous level. Is there
        # a case for mixins? Submodel approach seems decent, could repeat for
        # WithDeferredSubsystems, then Model layer is quite complete

        # TODO: add support for inheriting other models -- subclasses are
        # modifying/adding (no deletion? need mixins to pre-build pieces?) to parent
        # classes.  case 4
        # fields would need to combine _elements

        new_attrs = {}
        condor_attrs = {}
        for k, v in attrs.items():
            if cls.is_condor_attr(
                k,
                v,
            ):
                condor_attrs[k] = v
            else:
                new_attrs[k] = v

        # don't want model_name in kwargs at this point, needs to get popped?
        # or don't pass kwargs to super new...

        # TODO: I think all of these TODOs have been completed?
        # TODO I want ModelType to be responsible for popping `Optionns`, and maybe
        # `dynamic_link` -- only needed for user model declaration, if at all
        # TODO and I guess similarly, ModelTemplateType should be responsible for
        # popping placeholder (after injecting)
        # TODO how to maintain reserved words? what is injected/popped by subclasses
        # here? clashes are IO and reserved words
        # TODO only do check_attr_name for condor_attrs? no, need to make sure no
        # clashes occur...
        # TODO how to protect _meta? reserved words list... how to have each reserved
        # words list inherit from its parent?
        # TODO factor out (model) name processing...

        if cls.creating_base_class_for_inheritance(name):
            log.debug("creating base class for inheritance, pre super new")
            log.debug("cls.baseclass_for_inheritance=%s", cls.baseclass_for_inheritance)
            log.debug("%s, %s", name, bases)
            log.debug(
                "Any changes to keys? %s, %s",
                set(attrs.keys()) - set(new_attrs.keys()),
                condor_attrs,
            )
            # should not need the condor_attrs processing TODO
            # is the user_model_metaclass and user_model_baseclass always follow the
            # drop Template, drop Type pattern?

        # TODO check that bases was never modified by pre_super_new??
        new_cls = super().__new__(
            cls,
            name,
            bases,
            new_attrs,
            **kwargs,
        )
        new_cls._meta = dc.replace(attrs.meta)
        # TODO why did process_fields happen before?
        cls.process_fields(new_cls)

        for k, v in condor_attrs.items():
            # TODO: is this actually a new_cls classmethod? should that be the same as a
            # cls.method?
            log.debug(f"processing {k}={v} on {new_cls}")
            cls.process_condor_attr(k, v, new_cls)

        if cls.creating_base_class_for_inheritance(name):
            log.debug("creating base class for inheritance, post super new")
            log.debug("%s, %s", name, bases)
            cls.baseclass_for_inheritance = new_cls

        return new_cls

    @classmethod
    def process_fields(cls, new_cls):
        # perform as much processing as possible before caller super().__new__
        # by building up super_attrs from attrs; some operations need to operate on the
        # constructed class, like binding fields and submodels
        # before calling super new, process fields
        for field in new_cls._meta.all_fields:
            field.process_field()

        # elements from input and input fields are added directly to model
        # previously, all fields were  "finalized" by creating dataclass
        # ODESystem has no implementation and not all the fields should be finalized
        # Events may create new parameters that are canonical to the ODESystem model,
        # and should be not be finalized until a trajectory analysis, which has its own
        # copy. state must be fully defined by ODEsystem and can/should be finalized.
        # What about output? Should fields have an optional skip_finalize or something
        # that for when an submodel class may modify it so don't finalize? And is that
        # the only time it would come up?
        # TODO: review this

        for input_field in new_cls._meta.input_fields:
            for in_element in input_field:
                in_name = in_element.name
                check_attr_name(in_name, in_element, new_cls)
                setattr(new_cls, in_name, in_element)
                new_cls._meta.input_names.append(in_name)

        for _ in new_cls._meta.internal_fields:
            pass

        for output_field in new_cls._meta.output_fields:
            for out_element in output_field:
                out_name = out_element.name
                check_attr_name(out_name, out_element, new_cls)
                setattr(new_cls, out_name, out_element)
                new_cls._meta.output_names.append(out_name)

        for field in new_cls._meta.all_fields:
            setattr(new_cls, field._name, field)

    @classmethod
    def process_condor_attr(cls, attr_name, attr_val, new_cls):
        """after standard python __new__, call this for any attribute that passed the
        is_condor_attribute check"""
        meta = new_cls._meta
        pass_attr = True
        if isinstance(attr_val, BaseElement):
            pass
        if isinstance(attr_val, backend.symbol_class):
            element = new_cls._meta.backend_repr_elements.get(attr_val, None)
            if element is not None:
                log.debug(
                    "element %s=%s was found and is being passed", attr_name, attr_val
                )
                attr_val = element
            else:
                log.debug(
                    "symbol %s=%s was NOT found, NOT passing", attr_name, attr_val
                )
                pass_attr = False
        # not sure if this should really be Base or just Model? Or maybe submodel by
        # now...
        # TODO check isinstance type
        if (
            isinstance(attr_val, BaseModelType)
            and attr_val.primary in bases
            and attr_val in attr_val.primary._meta.submodels
        ):
            # handle submodel classes below, don't add to super
            return
        if isinstance(attr_val, Field):
            attr_val.bind(attr_name, new_cls)

        if meta.inherited_items.get(attr_name, None) is attr_val:
            log.debug("not passing %s because it was inherited", attr_name)
            # templates should not pass on placeholder because it is inherited, but
            # models should pass on remaining fields even thought hey are inherited!
            pass_attr = False

        if attr_name == "Options":
            new_cls.Options = attr_val

        if attr_name in cls.reserved_words:
            log.debug(
                # raise ValueError(
                "NOT passing on %s because it is a reserved word for %s",
                attr_name,
                cls,
                # )
            )
            pass_attr = False

        if pass_attr:
            check_attr_name(attr_name, attr_val, new_cls)
            setattr(new_cls, attr_name, attr_val)

    def register(cls, subclass):
        cls._meta.subclasses.append(subclass)


def check_attr_name(attr_name, attr_val, new_cls):
    if attr_name in new_cls.__class__.reserved_words:
        msg = (
            f"Cannot assign attribute {attr_name}={attr_val} because {attr_name} is a "
            "reserved word"
        )
        raise NameError(msg)
    existing_attr = new_cls.__dict__.get(attr_name, None)
    if existing_attr is not None and attr_val is not existing_attr:
        if isinstance(existing_attr, BaseElement):
            compare_attr_existing = existing_attr.backend_repr
        else:
            compare_attr_existing = existing_attr

        if isinstance(attr_val, BaseElement):
            compare_attr_new = attr_val.backend_repr
        else:
            compare_attr_new = attr_val

        if isinstance(compare_attr_existing, backend.symbol_class):
            if backend.symbol_is(compare_attr_new, compare_attr_existing):
                return
        elif compare_attr_new is compare_attr_existing:
            return

        msg = (
            f"Cannot assign attribute {attr_name}={attr_val} because {attr_name} is "
            f"already set to {getattr(new_cls, attr_name)}"
        )
        raise NameError(msg)
    return


# TODO stub for creating MetaData for ModelTemplate's
# @dataclass
# class ModelTemplateMetaData(BaseModelMetaData):
#    model_metaclass: object = None


class ModelTemplateType(BaseModelType):
    """Define a Model Template by subclassing Model, creating field types, and writing
    an implementation.
    """

    # metadata_class = SubmodelMetaData
    user_model_metaclass = None
    user_model_baseclass = None
    reserved_words = BaseModelType.reserved_words + [
        "placeholder",
    ]

    def __init_subclass__(cls, **kwargs):
        cls.user_model_metaclass = None
        cls.user_model_baseclass = None
        super().__init_subclass__(**kwargs)

    @classmethod
    def is_user_model(cls, bases):
        return (
            cls.baseclass_for_inheritance is not None
            and cls.baseclass_for_inheritance not in bases
        )

    @classmethod
    def is_template(cls, bases):
        return (
            cls.baseclass_for_inheritance is not None
            and cls.baseclass_for_inheritance in bases
        )

    @classmethod
    def is_condor_attr(cls, k, v):
        return super().is_condor_attr(k, v)

    @classmethod
    def __prepare__(
        cls, model_name, bases, as_template=False, model_metaclass=None, **kwargs
    ):
        if model_metaclass is not None:
            log.debug("prcessing for model metaclass asssigned, prepare")
        if cls.is_user_model(bases) and not as_template:
            log.debug(
                "dispatch __prepare__ for user model %s, %s",
                cls.user_model_metaclass,
                cls.user_model_baseclass,
            )
            return bases[0].user_model_metaclass.__prepare__(
                model_name, bases + (cls.user_model_baseclass,), **kwargs
            )
        else:
            if (
                (meta := kwargs.pop("meta", None)) is None
                and bases
                and (
                    user_model_metclass := getattr(
                        bases[0], "user_model_metaclass", None
                    )
                )
                is not None
            ):
                meta = user_model_metclass.metadata_class(model_name=model_name)
            # actually creating a model template, TODO at some point tap into
            # inheritance tree for now nothing fails this check
            if as_template:
                bases_mro = tuple()
                for base in bases:
                    bases_mro += base.__mro__
                if Model in bases_mro or ModelTemplate in bases_mro:
                    pass
                else:
                    breakpoint()
            return super().__prepare__(model_name, bases, meta=meta, **kwargs)

    @classmethod
    def prepare_populate(cls, cls_dict, bases):
        if not cls.creating_base_class_for_inheritance(cls_dict.meta.model_name):
            log.debug("injecting placeholder on %s", cls_dict.meta.model_name)
            cls_dict["placeholder"] = WithDefaultField(
                Direction.internal,
                name="placeholder",
                model_name=cls_dict.meta.model_name,
            )

        super().prepare_populate(cls_dict, bases)

    @classmethod
    def process_condor_attr(cls, attr_name, attr_val, new_cls):
        pass_super = True
        if attr_name == "placeholder":
            new_cls.placeholder = attr_val
            pass_super = False

        if attr_name in new_cls._meta.inherited_items:
            pass_super = False
            log.debug("template not passing on, %s=%s", attr_name, attr_val)

        if pass_super:
            log.debug("template IS passing on %s=%s", attr_name, attr_val)
            super().process_condor_attr(attr_name, attr_val, new_cls)

    def __new__(
        cls, model_name, bases, attrs, as_template=False, model_metaclass=None, **kwargs
    ):
        if cls.is_user_model(bases) and not as_template:
            log.debug(
                "dispatch __new__ for user model %s, %s, %s",
                model_name,
                cls.user_model_metaclass,
                cls.user_model_baseclass,
            )
            # user model: inject the baseclass for inheritance to bases and call the
            # user metaclass
            user_model = bases[0].user_model_metaclass(
                model_name, bases + (cls.user_model_baseclass,), attrs, **kwargs
            )
            return user_model

        if cls.creating_base_class_for_inheritance(model_name):
            immediate_return = True
        else:
            immediate_return = False

        if as_template:  # and False:
            attrs.meta.inherited_items.update(attrs.meta.user_set)
            attrs.meta.user_set = attrs.meta.inherited_items
            attrs.meta.inherited_items = {}

        new_cls = super().__new__(cls, model_name, bases, attrs, **kwargs)

        if immediate_return:
            return new_cls

        if (
            model_metaclass is None
            and bases
            and bases[0] not in (ModelTemplate, SubmodelTemplate)
        ):
            # if not over-writing model_metaclass and bases is not built-in (so it may
            # have a real, new model metaclass) then use it
            log.debug("prcessing for inheriting model metaclass asssigned, new")
            set_model_metaclass = bases[0].user_model_metaclass
        else:
            # otherwise, just treat as input
            set_model_metaclass = model_metaclass
            log.debug("prcessing for model metaclass asssigned, new")

        if set_model_metaclass is not None:
            new_cls.user_model_metaclass = set_model_metaclass

        if model_metaclass is not None:
            # if over-writing model metaclass, this is the clas for inheritance
            model_metaclass.baseclass_for_inheritance = new_cls

        if as_template:
            impl = ModelType.get_implementation_class(new_cls)
            if impl is not None:
                setattr(
                    implementations,
                    new_cls.__name__,
                    impl,
                )

        return new_cls

    def type_kwargs(cls, meta):
        return {}


class ModelTemplate(metaclass=ModelTemplateType):
    """A baseclass for defining a Model Template. Eventually needs an implementation;

    injects a `placeholder` field which is processed by models
    has a default, if nan --> imply value must be provided. if None --> a dummy
    variable. Should fail if set
    """

    @classmethod
    def extend_template(
        cls, new_name="", new_meta=None, new_meta_kwargs=None, **kwargs
    ):
        if not new_name:
            new_name = cls.__name__

        if new_meta is None:
            if new_meta_kwargs is None:
                new_meta_kwargs = {}
            new_meta_kwargs.update(model_name=new_name, template=cls)
            new_meta = cls._meta.copy_update(**new_meta_kwargs)
            if new_meta.template is not cls:
                breakpoint()
                log.debug("did not overwrite template")
        elif isinstance(new_meta, BaseModelMetaData):
            if new_meta_kwargs is not None:
                raise TypeError()
        else:
            raise TypeError()

        if new_meta.template is not cls:
            breakpoint()
            raise ValueError

        type_kwargs = cls.type_kwargs(new_meta)
        new_dict = cls.__prepare__(new_name, cls.__bases__, **type_kwargs)
        new_placeholder_field = new_dict["placeholder"]
        if new_name == "TrajectoryAnalysis":
            log.debug("figuring out extend_template %s, %s", cls, new_name)
            pass
        for k, v in cls._meta.user_set.items():
            processed_v = getattr(cls, k, None)
            if (
                isinstance(processed_v, BaseElement)
                and processed_v.field_type is cls.placeholder
            ):
                new_processed_v = processed_v.copy_to_field(new_placeholder_field)
                new_dict[k] = new_processed_v.backend_repr
                continue
            if isinstance(v, Field):
                cls.inherit_field(v, new_dict)
            else:
                new_dict[k] = v

        extended_template = cls.__class__(
            new_name, cls.__bases__, new_dict, **type_kwargs
        )
        # TODO: don't love that I am doing this manually; should these have just been
        # put in some type of meta modeldata?
        extended_template.user_model_metaclass = cls.user_model_metaclass
        extended_template.user_model_baseclass = cls.user_model_baseclass
        if extended_template._meta.template is not cls:
            # TODO not sure why template didnt get set correctly even though I'm passing
            # it in to new_meta_kwargs...
            log.debug("did not overwrite template %s", cls)
        extended_template._meta.template = cls
        return extended_template


class ModelType(BaseModelType):
    """Type class for user models"""

    reserved_words = BaseModelType.reserved_words + [
        "Options",  # should pass on, used to mark condor_attr
        "options_dict",  # bound values of Options
        "Casadi",  # deprecated Options name
        "dynamic_link",  # raise error on user assignment
        "name",
        "model_name",
    ]

    def __repr__(cls):
        if cls.__bases__ == (object,):
            return f"<{cls.__name__}>"
        if cls.__bases__:
            return f"<{cls._meta.template.__name__}: {cls.__name__}>"

    @classmethod
    def __prepare__(
        cls, model_name, bases, bind_embedded_models=True, name="", meta=None, **kwargs
    ):
        if name:
            model_name = name
        if meta is None:
            meta = cls.metadata_class(
                model_name=model_name, bind_embedded_models=bind_embedded_models
            )
        elif meta.model_name != model_name:
            log.debug(
                "updated meta's model_name from %s to %s",
                meta.model_name,
                model_name,
            )
            meta.model_name = model_name
        log.debug(
            "ModelType.__prepare__(model_name=%s, bases=%s, **%s), meta=%s",
            model_name,
            bases,
            kwargs,
            meta,
        )

        return super().__prepare__(model_name, bases, meta=meta, **kwargs)

    @classmethod
    def prepare_populate(cls, cls_dict, bases):
        if not cls.creating_base_class_for_inheritance(cls_dict.meta.model_name):
            log.debug("injecting dynamic_link on %s", cls_dict.meta.model_name)
            cls_dict["dynamic_link"] = DynamicLink(cls_dict)

        super().prepare_populate(cls_dict, bases)

    @classmethod
    def placeholder_names(cls, new_cls):
        template = new_cls._meta.template
        if hasattr(template, "placeholder"):
            return template.placeholder.list_of("name")
        else:
            return []

    @classmethod
    def process_condor_attr(cls, attr_name, attr_val, new_cls):
        pass_super = True
        if isinstance(attr_val, (backend.symbol_class, BaseElement)):
            log.debug(
                "ModelType is checking a backend_repr... %s, %s, %s, %s",
                attr_name,
                attr_val,
                cls,
                new_cls,
            )
            if cls.is_user_model(new_cls.__bases__):
                pass

            if attr_name in cls.placeholder_names(new_cls):
                setattr(new_cls, attr_name, attr_val)
                pass_super = False

        if pass_super:
            super().process_condor_attr(attr_name, attr_val, new_cls)

    def __new__(
        cls, model_name, bases, attrs, bind_embedded_models=True, name="", **kwargs
    ):
        if not name:
            name = model_name

        if cls.creating_base_class_for_inheritance(name):
            super_bases = bases
            bases_mro = bases
        else:
            bases_mro = tuple()
            for base in bases:
                bases_mro += base.__mro__
            if Model in bases_mro or ModelTemplate in bases_mro:
                use_bases = bases
            else:
                # some type of model inheritance, TODO I think at some point need to
                # figure out how to thook into the inheritance tree. For now, just
                # use Model
                for base in bases:
                    template = base.__class__
                    if hasattr(template, "baseclass_for_inheritance"):
                        # first_base = template.baseclass_for_inheritance
                        # might be useful to get this?
                        break
                use_bases = (None, Model)

            super_bases = use_bases[1:]

        new_cls = super().__new__(cls, name, super_bases, attrs, **kwargs)

        if not cls.is_user_model(bases_mro):
            return new_cls
        # proceeding for user models, use bases_mro to support deep inheritance

        # update docstring
        orig_doc = attrs.get("__doc__", "")
        # process docstring / add simple equation
        lhs_doc = ", ".join([out_name for out_name in new_cls._meta.output_names])
        arg_doc = ", ".join([arg_name for arg_name in new_cls._meta.input_names])
        new_cls.__doc__ = "\n".join([orig_doc, f"    {lhs_doc} = {name}({arg_doc})"])

        # extend submodel templates
        for base in new_cls._meta.template.__mro__[:-2]:
            for submodel in base._meta.submodels:
                extended_submodel = submodel.extend_template(
                    new_meta_kwargs=dict(primary=new_cls)
                )
                new_cls._meta.submodels.append(extended_submodel)
                setattr(new_cls, submodel.__name__, extended_submodel)
                pass

        cls.inherit_template_methods(new_cls)

        cls.process_placeholders(new_cls, attrs)

        cls.bind_model_fields(new_cls, attrs)

        return new_cls

    @classmethod
    def inherit_template_methods(cls, new_cls):
        for base in new_cls._meta.template.__mro__[:-2][::-1]:
            for key, val in base._meta.user_set.items():
                # should
                if isinstance(val, classmethod):
                    log.debug(
                        "inheriting classmethod %s=%s from %s to %s",
                        key,
                        val,
                        new_cls._meta.template,
                        new_cls,
                    )
                    setattr(new_cls, key, val.__get__(None, new_cls))

                if not isinstance(val, Field) and callable(val):
                    log.debug(
                        "inheriting instance method %s=%s from %s to %s",
                        key,
                        val,
                        new_cls._meta.template,
                        new_cls,
                    )
                    setattr(new_cls, key, val)

    @classmethod
    def process_placeholders(cls, new_cls, attrs, placeholder_field=None):
        """perform placeholder substitution

        how to do an embedded model placeholder? use a deferred system to define? would
        user models need to substitue, or would same basic mechanism work?
        """
        # process placeholders
        # TODO -- if default = None, keep it as a dummy variable

        if placeholder_field is None:
            placeholder_field = new_cls._meta.template.placeholder
        placeholder_assignment_dict = placeholder_field.create_substitution_dict(
            new_cls, set_attr=True
        )

        for field in new_cls._meta.noninput_fields:
            for elem in field:
                if isinstance(elem.backend_repr, backend.symbol_class):
                    elem.backend_repr = backend.operators.substitute(
                        elem.backend_repr, placeholder_assignment_dict
                    )

    @classmethod
    def get_implementation_class(cls, new_cls):
        # process implementations
        implementation = getattr(cls, "implementation", None)
        if implementation is not None:
            return implementation
        if hasattr(new_cls, "Options"):
            implementation = getattr(new_cls.Options, "__implementation__", None)
            if implementation is not None:
                # inject so subclasses get this implementation
                # TODO: a better way? registration? etc?
                # implementation.__dict__[name] = implementation
                pass

            # TODO: other validation?
            # TODO: inherit options? No, defaults come from implementation itself
        else:
            new_cls.Options = type("Options", (), {})

        if implementation is None and new_cls._meta.template:
            implementation = getattr(
                implementations, new_cls._meta.template.__name__, None
            )

        return implementation

    @classmethod
    def bind_model_fields(cls, new_cls, attrs):
        implementation = cls.get_implementation_class(new_cls)

        if implementation is not None:
            cls.finalize_input_fields(new_cls)

            for field in attrs.meta.all_fields:
                if field not in attrs.meta.input_fields:
                    field.create_dataclass()
                field.bind_dataclass()

        for field in new_cls._meta.independent_fields:
            for element in field:
                setattr(new_cls, element.name, element)

    def finalize_input_fields(cls):
        for input_field in cls._meta.input_fields:
            input_field.create_dataclass()

    @classmethod
    def is_user_model(cls, bases):
        return (
            cls.baseclass_for_inheritance is not None
            and cls.baseclass_for_inheritance in bases
        )


class Model(metaclass=ModelType):
    """handles binding etc. for user models"""

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ["implementation"]}
        return d

    @staticmethod
    def function_call_to_fields(fields, *args, **kwargs):
        input_names = {elem.name: field for field in fields for elem in field}
        fields_kwargs = {field: dict() for field in fields}
        for (input_name, field), input_val in zip(input_names.items(), args):
            if input_name in kwargs:
                msg = (
                    f"Argument {input_name} has value {input_val} from args and "
                    f"{kwargs[input_name]} from kwargs"
                )
                raise ValueError(msg)
            fields_kwargs[field][input_name] = input_val

        missing_args = []
        extra_args = []

        for input_name, input_val in kwargs.items():
            if input_name not in input_names:
                # TODO: skip this one with a flag?
                extra_args.append(input_name)
                continue
            fields_kwargs[input_names[input_name]][input_name] = input_val

        for input_name, field in input_names.items():
            field_kwargs = fields_kwargs[field]
            if (input_value := field_kwargs.get(input_name, None)) is None:
                missing_args.append(input_name)
            elif isinstance(input_value, BaseElement):
                field_kwargs[input_name] = input_value.backend_repr

        if missing_args or extra_args:
            error_message = f"While calling {field._model}, "
            if extra_args:
                error_message += f"recieved extra arguments: {extra_args}"
            if extra_args and missing_args:
                error_message += " and "
            if missing_args:
                error_message += f"missing arguments: {missing_args}"
            raise ValueError(error_message)

        # TODO: check bounds on model inputs?
        # pack into dot-able storage, over-writting fields and elements

        out_fields = [field._dataclass(**fields_kwargs[field]) for field in fields]
        return out_fields

    def __init__(self, *args, name="", **kwargs):
        cls = self.__class__
        self.name = name

        # bind *args and **kwargs to to appropriate signature
        # TODO: is there a better way to do this?
        # yes, can refactor (including from_values, will need to do this for the
        # trajectory analysis convenience models, etc.
        # OR just get rid of *args, only accept by kwarg and do it how
        # OptimizationProblem.from_values currently does it

        self.bind_input_fields(*args, **kwargs)

        implementation_class = cls.get_implementation_class(cls)
        self.implementation = implementation_class(self)
        # self, list(self.input_kwargs.values())
        # )

        # generally implementations are responsible for binding computed values.
        # implementations know about models, models don't know about implementations
        if False:
            self.output_kwargs = {
                out_name: getattr(self, out_name)
                for field in cls._meta.output_fields
                for out_name in field.list_of("name")
            }

        self.bind_embedded_models()

    def bind_input_fields(self, *args, **kwargs):
        cls = self.__class__
        self.input_kwargs = {}
        bound_input_fields = cls.function_call_to_fields(
            cls._meta.input_fields, *args, **kwargs
        )

        for field_dc in bound_input_fields:
            self.bind_field(field_dc, symbols_to_instance=True)
            self.input_kwargs.update(field_dc.asdict())

    def bind_field(self, dataclass, symbols_to_instance=True):
        if symbols_to_instance:
            for k, v in dataclass.asdict().items():
                setattr(self, k, v)
        setattr(self, dataclass.field._name, dataclass)
        return dataclass

    @staticmethod
    def create_bound_field_dataclass(field, values, wrap=True):
        if wrap:
            values = backend.wrap(field, values)
        dataclass_kwarg = {elem_name: val in zip(field.list_of("name"), values)}
        return field._dataclass(**dataclass_kwarg)

    def __iter__(self):
        cls = self.__class__
        for output_name in cls._meta.output_names:
            yield getattr(self, output_name)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: "
            + ", ".join([f"{k}={v}" for k, v in self.input_kwargs.items()])
            + ">"
        )

    def bind_embedded_models(self):
        # TODO: how to have models cache previous results so this is always free?
        # Can imagine a parent model with multiple instances of the exact same
        # sub-model called with different parameters. Would need to memoize at least
        # that many calls, possibly more.

        # if model instance created with `name` attribute, result will be bound to that
        # name not the assigned name, but assigned name can still be used during model
        # definition

        model = self.__class__
        if not model._meta.bind_embedded_models:
            return

        model_assignments = backend.SymbolCompatibleDict()

        fields = [
            field
            for field in model._meta.input_fields + model._meta.output_fields
            if isinstance(field, FreeField)
        ]
        for field in fields:
            model_instance_field = getattr(self, field._name)
            model_instance_field_dict = dc.asdict(model_instance_field)
            model_assignments.update(
                {
                    elem.backend_repr: val
                    for elem, val in zip(field, model_instance_field_dict.values())
                }
            )

        for (
            embedded_model_ref_name,
            embedded_model_instance,
        ) in model._meta.embedded_models.items():
            embedded_model = embedded_model_instance.__class__
            bound_embedded_model = embedded_model.__new__(embedded_model)

            bound_embedded_model.implementation = embedded_model_instance.implementation
            embedded_model_instance.bind_input_as_embedded(
                self,
                bound_embedded_model,
                model_assignments,
            )

            bound_embedded_model.implementation(bound_embedded_model)

            setattr(self, embedded_model_ref_name, bound_embedded_model)

            model_assignments.update(
                embedded_model_instance.bind_output_as_embedded(
                    self,
                    bound_embedded_model,
                )
            )

            bound_embedded_model.bind_embedded_models()

    def bind_input_as_embedded(
        self,
        parent_instance,
        bound_embedded_model,
        model_assignments,
    ):
        embedded_model = self.__class__
        embedded_model_kwargs = {}
        for field in embedded_model._meta.input_fields:
            bound_field = getattr(self, field._name)
            bound_field_dict = dc.asdict(bound_field)
            for k, v in bound_field_dict.items():
                if not isinstance(v, backend.symbol_class):
                    embedded_model_kwargs[k] = v
                else:
                    value_found = False
                    for kk, vv in model_assignments.items():  # noqa: B007
                        if backend.symbol_is(v, kk):
                            value_found = True
                            break
                    if value_found:
                        embedded_model_kwargs[k] = vv
                    else:
                        symbols_in_v = backend.symbols_in(v)
                        v_kwargs = {
                            symbol: model_assignments[symbol] for symbol in symbols_in_v
                        }
                        embedded_model_kwargs[k] = backend.evalf(v, v_kwargs)

                    # not working because python is checking equality of stuff??
                    # model_assignments[v] = embedded_model_kwargs[k]

            # call alternate method, it's pretty small -- just do it here.
            # pattern is similar to from_values, etc. so maybe dry it up but this is
            # pretty small.
            # bound_embedded_model.bind_input_fields(**embedded_model_kwargs)
        bound_embedded_model.bind_input_fields(**embedded_model_kwargs)

    def bind_output_as_embedded(self, parent_instance, bound_embedded_model):
        assignment_updates = {}
        for field in self._meta.output_fields:
            sym_bound_field = getattr(self, field._name)
            sym_bound_field_dict = dc.asdict(sym_bound_field)

            ran_bound_field = getattr(bound_embedded_model, field._name)
            ran_bound_field_dict = dc.asdict(ran_bound_field)

            assignment_updates.update(
                {
                    sym_val: ran_val
                    for sym_val, ran_val in zip(
                        sym_bound_field_dict.values(), ran_bound_field_dict.values()
                    )
                    if isinstance(sym_val, backend.symbol_class)
                }
            )
        return assignment_updates


ModelTemplateType.user_model_metaclass = ModelType
ModelTemplateType.user_model_baseclass = Model

"""
Do we need class inheritance? "as_template" flag should suffice

Case 1: defining a new model template -- very fiew fields, just copy and paste;
manipulate and re-use implementaitons etc

Case 2: defining a new user model -- use sub-models or functions that perform
declarative operations for you? Need to think about this, I guess it should be possible
to support but probably better performance for functions. Could only be the same type??
I think relies on a get_or_create so variables can be re-used. Or maybe it really needs
substitution mechanism from placeholders which could be fine, this pattern came from
inheriting fields...

class inheritance would be a mechanism to translate ODE to DAE -- seems useful.

Case 3: Submodel version of same...
Not sure if it's a true inner model, but definitely need some capability to re-use both
Model and ModelTemplate on different "outer" model/template.

rename:
submodel --> embedded_model, generally no back reference
innermodel --> submodel, so inner_to is supermodel? parent? need to distinguish from
assembly relationship, I think? although I did wonder if assembly is just a special type
of submodel -- in both cases, want to generalize so templates can handle multiple
sub-types? Thinking of LinCov and adding noise to event, but maybe I only want to
support ODE vs DAE which is slightly different?

new gascon's library of components will need something like condor-flight's
"placeholder" field. Is this a field that is always useful for building library's? and
the metatype that goes through the output fields and does the substitutions when users
create their version? -- this is another *type* of declaration that needs a name. It's
more than a model template, but not yet a user model.

and, is it always used in assembly models? condor flight is a very structured assembly
-- planet (or really, baricenter/ central body), vehicles that must have planet parent,
then event/mode on vehicles... seems like there may be a relationship btwn placeholder
and assembly.

also, simulate "freezes" the assembly, which has been extremely useful for free drift
stuff, may be a nice mechanism for gasp missions, although maybe the "segment assembly"
and "operator" approach could be even better??

I guess we can just expect to provide class methods on assembly models that perform
symbolic operations on what currently exists? I think at that point, placeholders would
have been substituted for their values or appropriate field elements; so AssemblyModel
type can provide utility methods for tree traversal but users just define their own
thing. Maybe it needs to get wrapped in something like a model? not so different from
Vehicle's solve -

Or an "Airplane" submodel that acts on root component (maybe an OML body, ? maybe user
choice) which freezes thing, then Airplane has to have inputs bound and has the instance
methods for operating? But actually something like aero and propulsion still need to
create something like an explicit system (maybe it would be unique for gascon, something
like "dynamic/performance component" or something).

Maybe "Design" off of the root of the node? And that's what gets named by a particular
airplane? And root is just body. And then can even tree off the design for the
performance? Point performance modules, a "mission" is a path on the tree from design,
ground roll etc. can branch from mission to do OEI, etc.


Re: simulate to freeze:
Is it more efficient to have each free drift go through the whole sim, instead of saving
the state and re-initializing each segment separately? For RPOD trajectory design,
perhaps that would be a time when forward mode would be better... and is
forward-over-adjoint ALWAYS better? or is there a scenario in which forward-over-forward
(directional) is better?


both InnerModel and Model have the same 3 cases: core, template, user; Model "knows"
about InnerModel -- can we fix that?
Is there a better way to handle the 3 cases, or is it acceptable that the same metaclass
has to do all 3?

There are possibly 2 more: Model with placeholder (LibraryModel??), and AssemblyModel.
It sould be nice if all 3 sub-types could be mixed-and-matched.

Also, want a singleton placeholder element or descriptor -- basically a placeholder
field on models where users aren't adding elements. could be a detached field in Condor
that template makers can use? Actually, the use cases I'm thinking of
(OptimizationProblem.objective, Gascon.OMLComponent.Aero.{CL, CD},
Gascon.Propulsor.Propulsion.thrust, ) are outputs with
defaults -- its own field.

what about tf (I guess could be an expression)

submodels from atmosphere etc models from Condor-Flight?


placeholders: elements defined by library creators to allow user inputs; condor provides
substitution mechanisms, etc. ~ expected uer input

submodels are mdoels that don't make sense w/o their parent, maybe add configuration
(like singleton). inner_to arg becomes modifying? because submodels only exist to modify
their parent?  maybe submodel modifies its superior?


A user Model with a parent, a ModelTemplate with a parent is a submodel
 (possibly only Assembly models will allow user model to set a parent)

assemblies make sense on their own, so user must define relationship, and be able to
attach/be adopted/assign/etc. after model definition. -- hook for symbolic processing at
attachment?
are assmeblies just a special type for library makers to define useful datastructures?
need a submodel/type to actually operate on them and do computation...not sure what an
assembly of (mix and match) explicit systems, algebraic, etc would even be!
Actually, if computation is always on some end-point submodel, can do all processing
htere? Yes, assembly components are JUST a datastructure ot define inputs, it solves the
the sensor model problem from LinCov demo to automatically handle creating parameters, a
few things about accessing namespace, etc.

flags/controls to conntrol inheritance:
"as_template" flag -- dry way to create new but related templates similar to user model
inheritance
parent relationship inheritance: what gets shared vs copied vs. ??
copied --> computation/end-point submodel

TrajectoryAnalysis should get extra flags for keep/skip events and modes




"""


@dc.dataclass
class SubmodelMetaData(BaseModelMetaData):
    # primary only needed for submodels, maybe also subclasses?
    primary: object = None
    copy_fields: list = dc.field(default_factory=list)
    copy_embedded_models: bool = True
    subclasses: list = dc.field(default_factory=list)


class SubmodelTemplateType(ModelTemplateType):
    metadata_class = SubmodelMetaData

    def type_kwargs(cls, meta):
        return dict(
            primary=meta.primary,
            copy_fields=meta.copy_fields,
            copy_embedded_models=meta.copy_embedded_models,
        )

    @classmethod
    def inherit_field(cls, field, cls_dict):
        v_init_kwargs = field._init_kwargs.copy()
        for init_k, init_v in field._init_kwargs.items():
            did_update = False
            if isinstance(init_v, (list, tuple)):
                init_v_iterable = init_v
                originaly_iterable = True
            else:
                init_v_iterable = [init_v]
                originaly_iterable = False

            use_kwarg = []
            for v in init_v_iterable:
                if isinstance(v, Field):
                    # TODO: is it OK to always assume cls_dict has the proper reference
                    # injected
                    if v._name in cls_dict:
                        use_kwarg.append(cls_dict[v._name])
                        did_update = True
                        # v_init_kwargs[init_k] = cls_dict[v._name]
                    elif v._name in cls_dict.meta.primary.__dict__:
                        use_kwarg.append(cls_dict.meta.primary.__dict__[v._name])
                        did_update = True
                        # v_init_kwargs[init_k] = (
                        #     cls_dict.meta.primary.__dict__[v._name]
                        # )
            if not originaly_iterable and use_kwarg:
                use_kwarg = use_kwarg[0]
            if did_update:
                v_init_kwargs[init_k] = use_kwarg

        inherited_field = field.inherit(
            cls_dict.meta.model_name, field_type_name=field._name, **v_init_kwargs
        )
        cls_dict[field._name] = inherited_field
        cls_dict[field._name].prepare(cls_dict)

    @classmethod
    def __prepare__(
        cls,
        model_name,
        bases,
        primary=None,
        copy_fields=None,  # defaults to False, but need to copy from base if not set
        copy_embedded_models=None,
        **kwds,
    ):
        log.debug(
            "SubmodelTemplateType.__prepare__(model_name=%s, bases+%s, primary=%s, "
            "copy_fields=%s, copy_embedded_models=%s, **%s)",
            model_name,
            bases,
            primary,
            copy_fields,
            copy_embedded_models,
            kwds,
        )

        for base in bases:
            if isinstance(base, cls):
                base_meta = base._meta
                break

        if cls.is_user_model(bases):
            if primary is not None:
                msg = "User's submodels should not provide primary"
                raise TypeError(msg)
            if base_meta.primary is None:
                msg = (
                    "This shouldn't happen -- defining user submodel with submodel "
                    "template that is missing primary attribute"
                )
                raise TypeError(msg)
            primary = base_meta.primary

            if copy_fields is None:
                copy_fields = base_meta.copy_fields
            if copy_embedded_models is None:
                copy_embedded_models = base_meta.copy_embedded_models

        elif cls.creating_base_class_for_inheritance(model_name):
            # primary is allowed to be None
            if copy_fields is None:
                copy_fields = False
            if copy_embedded_models is None:
                copy_embedded_models = True
        else:
            # should extended models hit here??
            if primary is None:
                msg = "SubmodelTemplate requires primary"
                raise TypeError(msg)
            if copy_fields is None:
                copy_fields = False
            if copy_embedded_models is None:
                copy_embedded_models = True

        # TODO: starts to distinguish between user model metadata and template
        # metadata...
        if cls.is_user_model(bases):
            metadata_class = base.user_model_metaclass.metadata_class
        else:
            metadata_class = cls.metadata_class

        meta = metadata_class(
            model_name=model_name,
            copy_fields=copy_fields,
            primary=primary,
            copy_embedded_models=copy_embedded_models,
        )
        cls_dict = super().__prepare__(model_name, bases, meta=meta, **kwds)

        return cls_dict

    def __new__(
        cls,
        model_name,
        bases,
        attrs,
        primary=None,
        copy_fields=None,
        copy_embedded_models=None,
        **kwargs,
    ):
        new_cls = super().__new__(cls, model_name, bases[:], attrs, **kwargs)

        if (
            cls.baseclass_for_inheritance is not None
            and cls.baseclass_for_inheritance is not new_cls
        ):
            new_cls._meta.primary._meta.submodels.append(new_cls)
        return new_cls

    def __iter__(cls):
        yield from cls._meta.subclasses


class SubmodelTemplate(
    ModelTemplate,
    metaclass=SubmodelTemplateType,
    primary=None,
):
    pass


class SubmodelType(ModelType):
    metadata_class = SubmodelMetaData

    def __new__(cls, model_name, bases, attrs, **kwargs):
        new_cls = super().__new__(cls, model_name, bases, attrs, **kwargs)
        if cls.is_user_model(bases):
            new_cls._meta.template._meta.subclasses.append(new_cls)
        return new_cls

    @classmethod
    def placeholder_names(cls, new_cls):
        primary_template = new_cls._meta.primary._meta.template
        super_names = super().placeholder_names(new_cls)
        if hasattr(primary_template, "placeholder"):
            new_names = primary_template.placeholder.list_of("name")
        else:
            new_names = []
        return super_names + new_names

    @classmethod
    def prepare_item_from_primary(cls, cls_dict, attr_name, attr_val):
        if isinstance(attr_val, Field) and cls_dict.meta.copy_fields:
            if attr_val._direction == Direction.output:
                new_direction = Direction.internal
            else:
                new_direction = None
            cls.inherit_field(attr_val, cls_dict, new_direction)
            copied_field = cls_dict[attr_val._name]
            if isinstance(copied_field, FreeField):
                copied_field._elements = [sym for sym in attr_val]
            else:
                for elem in attr_val:
                    elem.copy_to_field(copied_field)
            # cls_dict[attr_name] = copied_field
            return
        if isinstance(attr_val, Model):
            if not cls_dict.meta.copy_embedded_models:
                return
            if cls_dict.meta.template._meta.model_name == "TrajectoryAnalysis":
                # breakpoint()
                pass
        cls_dict[attr_name] = attr_val

    @classmethod
    def prepare_populate(cls, cls_dict, bases):
        if cls.baseclass_for_inheritance is not None:
            primary_dict = {**cls_dict.meta.primary._meta.inherited_items}
            primary_dict.update(cls_dict.meta.primary._meta.user_set)
            for attr_name, attr_val in primary_dict.items():
                cls.prepare_item_from_primary(cls_dict, attr_name, attr_val)
        super().prepare_populate(cls_dict, bases)
        pass

    @classmethod
    def process_condor_attr(cls, attr_name, attr_val, new_cls):
        if isinstance(attr_val, Model) and not cls_dict.meta.copy_embedded_models:
            return

        if (
            isinstance(attr_val, Field)
            and not new_cls._meta.copy_fields
            and attr_val._inherits_from._model
            in (new_cls._meta.primary, new_cls._meta.primary._meta.template)
        ):
            return
        super().process_condor_attr(attr_name, attr_val, new_cls)

    @classmethod
    def is_user_model(cls, bases):
        if cls.baseclass_for_inheritance is None:
            return False
        if cls.baseclass_for_inheritance in bases:
            return True
        for base in bases:
            if base._meta.template is cls.baseclass_for_inheritance:
                return True
        return False


class Submodel(Model, metaclass=SubmodelType):
    pass


SubmodelTemplateType.user_model_metaclass = SubmodelType
SubmodelTemplateType.user_model_baseclass = Submodel
