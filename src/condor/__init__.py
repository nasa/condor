if False:
    import logging

    logging.getLogger("condor").setLevel(logging.DEBUG)
    logging.basicConfig()


from condor._version import __version__
from condor.conf import settings
from condor.contrib import *
from condor.fields import (AssignedField, BaseElement, BoundedAssignmentField,
                           Direction, Field, FreeElement, FreeField,
                           InitializedField, MatchedField,
                           TrajectoryOutputField, WithDefaultField)
from condor.models import Options

##################
"""
ModelTemplate flag for user_model (default True)
If false, can add placeholders, returns a template instead of model




ModelTemplate class creation
    creation of placeholder field
    that's it?

model_template class creation (e.g., AlgebraicSystem)
    inherit (copy) placeholder field
    creation of placeholder elements for users to fill-in
    creation of model template specific fields (e.g., implicit_output, residual)

extended template (for libraries):
    creation of elements that must get re-created for each user model
    specification of (some) parent template placeholder elements
    creation of new placeholder elements

    or is this just normal template creation?
    does this need to get "flattened" somehow?

submodel template:
    same as model/extended template and...
    specify primary
    and whether fields are
    copied or accessed
    --> attaching submodel template from template to user model is a special version of
    extension?? kwargs need to be optional, then just another layer of dispatch.

assembly/component model template:
    specify rules for acceptable child/parent (but also methods for later modification)
    template extension classes by default 

user assembly/component model creation:
    provide methods for operating on tree (maybe python multi-inherit)
    must take kwarg 

user model creation
    (condor-)inherit template-specific fields
    define model:
        creation of elements from template specific fields
        fill in (optional) values for placeholder elements
        embed models for sub-calculations
    create (extended) templates for submodels

    (python )inherit methods for binding, accessing IO data, etc.--not part of class creation


so __prepare__ has to handle all the accesible element and field injection (accessing/copying/etc), 
CondorClassDict has to handle assignment for elements/submodels/etc
__new__ is cleanup of standard class attributes before passing to type().__new__ and
finalization after -- I guess it's possible the finalization could happen in __init__ or
even __init_subclass__?

customize meta so only have relevant attriburtes
too hard to use different metaclasses for base/template/user ? times special types like
submodel, assemblycomponent.
if template inherits from base, user inherits from template, what happens? special types
can multi-inherit? 

need to be able to make it easy to build library components (extended templates)
how to "socket" something? e.g., quickly create a new assembly component like an
impulsive correction and implement __new__ finalize logic to modify ODE... similar to
condor-flight.
clear pattern (or even better, simple API to automate) 
if over-writing __new__, __prepare__, class_dict.__set_item__, is enough, that's OK. 
class_dict can use meta to access anything and then adding some generality to class_dict
filtering, maybe even adding little callback hooks or something. 

above sketch is still just datastructure. Need to trace through construction of a simple
model. 

LinCov is directional SGM, sone number of number to construct covariance matrix or
something


ModelTemplate:
provide placeholder, extend_template

Model:
provide a bunch of user-model methods

a template: inherit ModelTemplate
define fields and pre-loaded symbols (placeholders, independent variables, etc) and get
ready for inheritance to a user model -- 

a submodel template: inherit SubmodelTemplate
primary to a template, define fields and pre-loaded symbols

a user model: inherit from a template
"read" user-defined elements, including output from embeded models
inherit user-copies of fields, prepare to be be runnable -- substitue placeholder
elements with their subs (or defaults)
copy & extend any submodel templates

a user submodel: inherit from a user model's extended submodel template





currently...
__new__ is:
  some processing specific to submodel:
     don't add attrs that were originally from primary to submodel
     if it was an embedded model, also remove from meta embedded models list

  create _parent_name attr for __repr__

  filter attrs for:
      strip dynamic link
      options
      submodel template (which get added as a ref so users can create them)
      independent symbols (or their backend repr) that don't belong to a field bound to
        actual model -- processed by iterating over field's elements below

  process fields (and independentsymbol elements)
  - to update reference name which isn't known until now?
  - for input and output fields:
      name validation
      replace attribute name with element's name (from independent name assignment OR
          name kwarg on element creation)
      replace backend repr VALUE with element
      add to IO meta
  - dataclass creation for internal and  output fields (why not input fields??)

  modify docstring

  create class using super().__new__

  re-attach meta

  if primary (is submodel):
     improper separation, so needs to see if it is a submodel template or a valid
     submodel, and get added appropriately or not

  if not submodel template and not direct descendent of model, register?
  I think this is registering user models for each template

  final bind fields
  process inheriting submodel templates
  if implementaiton exists
      create input field dataclasses (why not earlier??)
      bind implementation
      bind dataclasses (attach qualname etc)

So a lot is happening that could happen in condorclassdict, new should be finalize-y
stuff. filter stuff from namespace that we want available for class declaration (user
convenience) but not on final class, call model, then finalize binding etc.

where does placeholder substitution go? this will specify where hooks for other
extensions go. And need to figure out assembly too

"""
