from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from daidepp.constants import *
from daidepp.keywords.daide_object import _DAIDEObject
from daidepp.keywords.keyword_utils import and_items, unit_dict


@dataclass
class Location:
    province: Union[str, Location]
    coast: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.province, Location):
            self.province = self.province.province

    def __str__(self) -> str:
        if self.coast:
            return f"({self.province} {self.coast})"
        return self.province


@dataclass
class Unit(_DAIDEObject):
    power: Power
    unit_type: UnitType
    location: Location

    def __str__(self):
        unit = unit_dict[self.unit_type]
        return f"{self.power}'s {unit} in {self.location} "


@dataclass
class HLD(_DAIDEObject):
    unit: Unit

    def __str__(self):
        return f"holding {self.unit} "


@dataclass
class MTO(_DAIDEObject):
    unit: Unit
    location: Location

    def __str__(self):
        return f"moving {self.unit} to {self.location} "


@dataclass
class SUP:
    supporting_unit: Unit
    supported_unit: Unit
    province_no_coast: Optional[ProvinceNoCoast] = None

    def __str__(self):
        if not self.province_no_coast:
            return f"using {self.supporting_unit} to support {self.supported_unit} "
        else:
            return f"using {self.supporting_unit} to support {self.supported_unit} moving into {self.province_no_coast} "


@dataclass
class CVY:
    convoying_unit: Unit
    convoyed_unit: Unit
    province: ProvinceNoCoast

    def __str__(self):
        return f"using {self.convoying_unit} to convoy {self.convoyed_unit} into {self.province} "


@dataclass
class MoveByCVY(_DAIDEObject):
    unit: Unit
    province: Location
    province_seas: List[Location]

    def __init__(self, unit, province, *province_seas):
        self.unit = unit
        self.province = province
        self.province_seas = province_seas

    def __str__(self):
        return (
            f"moving {self.unit} by convoy into {self.province} via "
            + and_items(list(map(lambda x: str(x), self.province_seas)))
        )


@dataclass
class RTO(_DAIDEObject):
    unit: Unit
    location: Location

    def __str__(self):
        return f"retreating {self.unit} to {self.location} "


@dataclass
class DSB(_DAIDEObject):
    unit: Unit

    def __str__(self):
        return f"disbanding {self.unit} "


@dataclass
class BLD(_DAIDEObject):
    unit: Unit

    def __str__(self):
        return f"building {self.unit} "


@dataclass
class REM(_DAIDEObject):
    unit: Unit

    def __str__(self):
        return f"removing {self.unit} "


@dataclass
class WVE(_DAIDEObject):
    power: Power

    def __str__(self):
        return f"waiving {self.power} "


@dataclass
class Turn(_DAIDEObject):
    season: Season
    year: int

    def __str__(self):
        return f"{self.season} {self.year} "


Retreat = Union[RTO, DSB]
Build = Union[BLD, REM, WVE]
Order = Union[
    HLD,
    MTO,
    SUP,
    CVY,
    MoveByCVY,
]
Command = Union[Order, Retreat, Build]
