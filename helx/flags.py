# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import dataclasses
from typing import Sequence, get_type_hints

import chex
from absl import flags
from absl.flags import DEFINE, _argument_parser, _flagvalues

from .spaces import Space, Continuous, Discrete


class ShapeParser(_argument_parser.ListParser):
    def parse(self, shape: chex.Shape):
        return tuple(map(int, super().parse(shape)))


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


class SpaceParser(_argument_parser.ListParser):
    def parse(self, space: Space):
        shape = tuple(map(int, super().parse(space.shape)))
        dtype = space.dtype
        if shape != (1,):
            space = Continuous(shape, dtype)
        else:
            space = Discrete(1)
        return (shape, space.dtype)


def DEFINE_space(  # pylint: disable=invalid-name,redefined-builtin
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
    parser = SpaceParser()
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
    elif type_ == Space:
        return DEFINE_space
    elif type_ in (list, Sequence, tuple):
        return flags.DEFINE_list
    else:
        raise ValueError("Type {} is not supported".format(type_))


def get_hparams_fields(type_):
    fields = get_type_hints(type_)
    if "obs_space" in fields:
        del fields["obs_space"]
    if "action_space" in fields:
        del fields["action_space"]
    return fields


def define_flags_from_hparams(type_):
    # get_type_hints to get precise types than annotations
    fields = get_hparams_fields(type_)
    if "obs_space" in fields:
        del fields["obs_space"]
    if "action_space" in fields:
        del fields["action_space"]

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


def hparams_from_flags(
    cls, obs_space: Space, action_space: Space, flags=flags.FLAGS, **kwargs
):
    hparams = {}
    for key, value in kwargs.items():
        hparams[key] = value
    for field in get_hparams_fields(cls):
        if field in kwargs:
            continue
        value = flags[field].value
        if value is None:
            raise ValueError("Flag {} is required".format(field))
        hparams[field] = value
    hparams["obs_space"] = obs_space
    hparams["action_space"] = action_space
    return cls(**hparams)
