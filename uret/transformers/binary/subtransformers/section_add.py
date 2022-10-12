from uret.transformers import SubTransformer
from uret.transformers.binary.subtransformers import binary_to_bytez

import lief
import random
from copy import deepcopy


class SectionAdd(SubTransformer):
    name = "SectionAdd"

    type_mapper = {
        "bss": lief.PE.SECTION_TYPES.BSS,
        "data": lief.PE.SECTION_TYPES.DATA,
        "export": lief.PE.SECTION_TYPES.EXPORT,
        "idata": lief.PE.SECTION_TYPES.IDATA,
        "relocation": lief.PE.SECTION_TYPES.RELOCATION,
        "resource": lief.PE.SECTION_TYPES.RESOURCE,
        "text": lief.PE.SECTION_TYPES.TEXT,
        "tls": lief.PE.SECTION_TYPES.TLS_,
        "unknown": lief.PE.SECTION_TYPES.UNKNOWN,
    }

    def __init__(self, length=[6, 12], types=None, transformation_range=None, seed=None, subtransformer_index=None):
        """
        Initialize a `SectionAdd` object. This object adds a new section of a certain type to the binary
        :param length: The maximum or  [lower, upper] bound in bytes of section length.
        :param types: A list of section type names.
        :param seed: Random seed value
        :param subtransformer_index: An int indicating where in the transformation record the subtransformer is defined. This is used to enforce "max # of transformations for this subtransformer" constraint.
        """

        if isinstance(length, list):
            self.min_val = length[0]
            self.max_val = length[1]
        else:
            self.min_val = 1
            self.max_val = length[1]

        if types is None:
            self.types = [
                lief.PE.SECTION_TYPES.BSS,
                lief.PE.SECTION_TYPES.DATA,
                lief.PE.SECTION_TYPES.EXPORT,
                lief.PE.SECTION_TYPES.IDATA,
                lief.PE.SECTION_TYPES.RELOCATION,
                lief.PE.SECTION_TYPES.RESOURCE,
                lief.PE.SECTION_TYPES.TEXT,
                lief.PE.SECTION_TYPES.TLS_,
                lief.PE.SECTION_TYPES.UNKNOWN,
            ]
        else:
            self.types = []
            for t in types:
                if t in type_mapper.keys():
                    self.types.append(type_mapper[t])
                else:
                    print(t + " is not a recognized type. Skipping")

        if len(self.types) == 0:
            raise ValueError("type list is empty")
        self.seed = seed
        self.subtransformer_index = subtransformer_index

        super(SectionAdd, self).__init__()

    def transform(self, x, transformation_record, transformation_value=None):
        """
        Add a new section to the binary
        :param x: Input Value
        :param transformation_record: Record of changes
        :param transformation_value: type of section to add
        :return: A transformed input and modified transformation record
        """
        random.seed(self.seed)
        binary = lief.PE.parse(list(x))

        section_name = "".join(chr(random.randrange(ord("."), ord("z"))) for _ in range(6))
        new_section = lief.PE.Section(section_name)

        if transformation_value is None:
            transformation_value = random.choice(self.types)

        # fill section with random stuff
        upper = random.randrange(256)
        content_length = random.randint(self.min_val, self.max_val)
        new_section.content = [random.randint(0, upper) for _ in range(content_length)]
        new_section.virtual_address = max([s.virtual_address + s.size for s in binary.sections])
        binary.add_section(new_section, transformation_value)

        new_x = binary_to_bytez(binary)

        # update the transformation record
        transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
        if self.subtransformer_index is not None:
            transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
        else:
            transformation_record["current_state"]["actions_taken"] += 1

        transformation_record["current_state"]["value"] = new_x

        return new_x, transformation_record

    def get_action_list(self):
        """
        Return a list of section types
        """
        return self.types

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by transformation_value can be performed on the current input.
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: If action is possible and the transformation value. The effect on the record is not used
        """
        return True, [transformation_value], None

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input.
        :param x: input_value

        :return: A list of transformation values
        """
        return self.types
