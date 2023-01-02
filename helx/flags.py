import dataclasses
from typing import Sequence, get_type_hints
from absl import flags
from absl.flags import _flagvalues, _argument_parser, DEFINE
import chex


class ShapeParser(_argument_parser.ListParser):
    def parse(self, arg):
        return tuple(map(int, super().parse(arg)))


def DEFINE_shape(  # pylint: disable=invalid-name,redefined-builtin
    name, default, help, flag_values=_flagvalues.FLAGS, required=False, **args
):
    """Registers a flag whose value is a comma-separated list of dimensions.

    The flag value is parsed with a CSV parser.

    It is the same as a ListParser, but the output is a tuple of integers.

    Args:
      name: str, the flag name.
      default: list|str|None, the default value of the flag.
      help: str, the help message.
      flag_values: :class:`FlagValues`, the FlagValues instance with which the
        flag will be registered. This should almost never need to be overridden.
      required: bool, is this a required flag. This must be used as a keyword
        argument.
      **args: Dictionary with extra keyword args that are passed to the
        ``Flag.__init__``.

    Returns:
      a handle to defined flag.
    """
    parser = ShapeParser()
    serializer = _argument_parser.CsvListSerializer(",")
    return DEFINE(
        parser, name, default, help, flag_values, serializer, required=required, **args
    )


def type_to_flag(type_):
    if type_ == bool:
        return flags.DEFINE_boolean
    elif type_ == int:
        return flags.DEFINE_integer
    elif type_ == float:
        return flags.DEFINE_float
    elif type_ == str:
        return flags.DEFINE_string
    elif type_ == chex.Shape:
        return DEFINE_shape
    elif type_ in (list, Sequence, tuple):
        return flags.DEFINE_list
    else:
        raise ValueError("Type {} is not supported".format(type_))


def define_flags_from_hparams(type_):
    # get_type_hints to get precise types than annotations
    fields = get_type_hints(type_)

    # we wrap the type in a dataclass to get the default values
    dataclass_fields = dataclasses.dataclass(type_).__dataclass_fields__

    for field in fields:
        default_value = dataclass_fields[field].default

        if default_value == dataclasses.MISSING:
            default_value = None

        define_fn = type_to_flag(fields[field])
        define_fn(
            name=field,
            default=default_value,
            help=str(type.__doc__),
            required=False,
        )
    return


def hparams_from_flags(type_, flags, **kwargs):
    hparams = {}
    for key, value in kwargs.items():
        hparams[key] = value
    for field in get_type_hints(type_):
        if field in kwargs:
            continue
        value = flags[field].value
        if value is None:
            raise ValueError("Flag {} is required".format(field))
        hparams[field] = value

    return type_(**hparams)
