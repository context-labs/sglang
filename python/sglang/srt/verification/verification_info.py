from enum import IntEnum, auto


class VerificationAlgorithm(IntEnum):
    NONE = auto()
    TOPLOC = auto()

    def is_none(self):
        return self == VerificationAlgorithm.NONE

    def is_toploc(self):
        return self == VerificationAlgorithm.TOPLOC

    @staticmethod
    def from_string(name: str):
        name_map = {
            "TOPLOC": VerificationAlgorithm.TOPLOC,
            None: VerificationAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
