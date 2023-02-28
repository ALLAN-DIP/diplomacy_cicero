from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from daidepp.constants import *
from daidepp.keywords.base_keywords import *
from daidepp.keywords.daide_object import _DAIDEObject
from daidepp.keywords.keyword_utils import and_items, or_items

@dataclass
class PCE(_DAIDEObject):
    powers: List[Power]

    def __init__(self, *powers):
        self.powers = powers
    

    def __str__(self):
        return "peace between " + and_items(self.powers)


@dataclass
class CCL(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"canceling \"{self.press_message}\" "


@dataclass
class TRY(_DAIDEObject):
    try_tokens: List[TryTokens]

    def __init__(self, *try_tokens):
        self.try_tokens = try_tokens
    

    def __str__(self):
        return "trying the following tokens: " + " ".join(self.try_tokens) + " "


@dataclass
class HUH(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"not understanding \"{self.press_message}\" "


@dataclass
class PRP(_DAIDEObject):
    arrangement: Arrangement
    power: str

    def __str__(self):
        return f"{self.power} propose {self.arrangement} "


@dataclass
class ALYVSS(_DAIDEObject):
    aly_powers: List[Power]
    vss_powers: List[Power]

    def __str__(self):
        return (
            "an ally of "
            + and_items(self.aly_powers)
            + "against "
            + and_items(self.vss_powers)
        )


@dataclass
class SLO(_DAIDEObject):
    power: Power

    def __str__(self):
        return f"{self.power} solo"


@dataclass
class NOT(_DAIDEObject):
    arrangement_qry: Union[Arrangement, QRY]

    def __str__(self):
        return f"not {self.arrangement_qry} "


@dataclass
class NAR(_DAIDEObject):
    arrangement: Arrangement

    def __str__(self):
        return f"NAR ( {self.arrangement} )"


@dataclass
class DRW(_DAIDEObject):
    powers: Optional[List[Power]] = ()

    def __init__(self, *powers):
        self.powers = powers
    

    def __str__(self):
        if self.powers:
            return and_items(self.powers) + "draw "
        else:
            return f"draw"


@dataclass
class YES(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"accepting \"{self.press_message}\" "


@dataclass
class REJ(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"rejecting \"{self.press_message}\" "


@dataclass
class BWX(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"refusing to answer to \"{self.press_message}\" "


@dataclass
class FCT(_DAIDEObject):
    arrangement_qry_not: Union[Arrangement, QRY, NOT]

    def __str__(self):
        return f"\"{self.arrangement_qry_not}\" is true "


@dataclass
class FRM(_DAIDEObject):
    frm_power: Power
    recv_powers: List[Power]
    message: Message

    def __str__(self):
        return (
            f"from {self.frm_power} to "
            + and_items(self.recv_powers)
            + f": \"{self.message}\" "
        )


@dataclass
class XDO(_DAIDEObject):
    order: Order

    def __str__(self):
        return f"an order {self.order} "


@dataclass
class DMZ(_DAIDEObject):
    powers: List[Power]
    provinces: List[Location]

    def __str__(self):
        return (
            and_items(self.powers)
            + "removing all units from, and not ordering to, supporting to, convoying to, retreating to, or building any units in "
            + and_items(list(map(lambda x: str(x), self.provinces)))
        )


@dataclass
class AND(_DAIDEObject):
    arrangements: List[Arrangement]

    def __init__(self, *arrangements):
        self.arrangements = arrangements
    

    def __str__(self):
        return and_items(self.arrangements)


@dataclass
class ORR(_DAIDEObject):
    arrangements: List[Arrangement]

    def __init__(self, *arrangements):
        self.arrangements = arrangements
    

    def __str__(self):
        return or_items(self.arrangements)


@dataclass
class PowerAndSupplyCenters:
    power: Power
    supply_centers: List[Location]  # Supply centers

    def __init__(self, power, *supply_centers):
        self.power = power
        self.supply_centers = supply_centers

    def __str__(self):
        return f"{self.power} to have " + and_items(list(map(lambda x: str(x), self.supply_centers)))


@dataclass
class SCD(_DAIDEObject):
    power_and_supply_centers: List[PowerAndSupplyCenters]

    def __init__(self, *power_and_supply_centers):
        self.power_and_supply_centers = power_and_supply_centers
    

    def __str__(self):
        pas_str = [str(pas) + " " for pas in self.power_and_supply_centers]
        return f"arranging supply centre distribution as follows: " + and_items(pas_str)


@dataclass
class OCC(_DAIDEObject):
    units: List[Unit]

    def __init__(self, *units):
        self.units = units
    

    def __str__(self):
        unit_str = [str(unit) for unit in self.units]
        return f"placing " + and_items(unit_str)


@dataclass
class CHO(_DAIDEObject):
    minimum: int
    maximum: int
    arrangements: List[Arrangement]

    def __init__(self, minimum, maximum, *arrangements):
        self.minimum = minimum
        self.maximum = maximum
        self.arrangements = arrangements
    

    def __str__(self):
        if self.minimum == self.maximum:
            return f"choosing {self.minimum} in " + and_items(self.arrangements)
        else:
            return f"choosing between {self.minimum} and {self.maximum} in " + and_items(self.arrangements)


@dataclass
class INS(_DAIDEObject):
    arrangement: Arrangement
    power: str

    def __str__(self):
        return f"{self.power} insist {self.arrangement} "


@dataclass
class QRY(_DAIDEObject):
    arrangement: Arrangement

    def __str__(self):
        return f"Is {self.arrangement} true? "


@dataclass
class THK(_DAIDEObject):
    arrangement_qry_not: Union[Arrangement, QRY, NOT, None]
    power: str

    def __str__(self):
        return f"{self.power} think {self.arrangement_qry_not} is true "


@dataclass
class IDK(_DAIDEObject):
    qry_exp_wht_prp_ins_sug: Union[QRY, EXP, WHT, PRP, INS, SUG]
    power: str

    def __str__(self):
        return f"{self.power} don't know about {self.qry_exp_wht_prp_ins_sug} "


@dataclass
class SUG(_DAIDEObject):
    arrangement: Arrangement
    power: str

    def __str__(self):
        return f"{self.power} suggest {self.arrangement} "


@dataclass
class WHT(_DAIDEObject):
    unit: Unit

    def __str__(self):
        return f"What do you think about {self.unit} ? "


@dataclass
class HOW(_DAIDEObject):
    province_power: Union[Location, Power]

    def __str__(self):
        return f"How do you think we should attack {self.province_power} ? "


@dataclass
class EXP(_DAIDEObject):
    turn: Turn
    message: Message
    power: str

    def __str__(self):
        return f"The explanation for what {self.power} did in {self.turn} is {self.message} "


@dataclass
class SRY(_DAIDEObject):
    exp: EXP

    def __str__(self):
        return f"I'm sorry about {self.exp} "


@dataclass
class FOR(_DAIDEObject):
    start_turn: Turn
    end_turn: Optional[Turn]
    arrangement: Arrangement

    def __str__(self):
        if not self.end_turn:
            return f"doing {self.arrangement} in {self.start_turn} "
        else:
            return f"doing {self.arrangement} from {self.start_turn} to {self.end_turn} "


@dataclass
class IFF(_DAIDEObject):
    arrangement: Arrangement
    press_message: PressMessage
    els_press_message: Optional[PressMessage] = None

    def __str__(self):
        if not self.els_press_message:
            return f"if {self.arrangement} then \"{self.press_message}\" "
        else:
            return f"if {self.arrangement} then \"{self.press_message}\" else \"{self.els_press_message}\" "


@dataclass
class XOY(_DAIDEObject):
    power_x: Power
    power_y: Power

    def __str__(self):
        return f"{self.power_x} owes {self.power_y} "


@dataclass
class YDO(_DAIDEObject):
    power: Power
    units: List[Unit]

    def __init__(self, power, *units):
        self.power = power
        self.units = units
    

    def __str__(self):
        unit_str = [str(unit) for unit in self.units]
        return f"giving {self.power} the control of" + and_items(unit_str)


@dataclass
class SND(_DAIDEObject):
    power: Power
    recv_powers: List[Power]
    message: Message

    def __str__(self):

        return (
            f"{self.power} sending {self.message} to "
            + and_items(self.recv_powers)
        )


@dataclass
class FWD(_DAIDEObject):
    powers: List[Power]
    power_1: Power
    power_2: Power

    def __str__(self):
        return (
            f"forwarding to {self.power_2} if {self.power_1} receives message from "
            + and_items(self.powers)
        )


@dataclass
class BCC(_DAIDEObject):
    power_1: Power
    powers: List[Power]
    power_2: Power

    def __str__(self):
        return (
            f"forwarding to {self.power_2} if {self.power_1} sends message to "
            + and_items(self.powers)
        )


@dataclass
class WHY(_DAIDEObject):
    fct_thk_prp_ins: Union[FCT, THK, PRP, INS]

    def __str__(self):
        return f"Why do you believe \"{self.fct_thk_prp_ins}\" ? "


@dataclass
class POB(_DAIDEObject):
    why: WHY

    def __str__(self):
        return f"answering {self.why} : the position on the board, or the previous moves, suggests/implies it "


@dataclass
class UHY(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"I'm unhappy that \"{self.press_message}\" "


@dataclass
class HPY(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"I'm  happy that \"{self.press_message}\" "


@dataclass
class ANG(_DAIDEObject):
    press_message: PressMessage

    def __str__(self):
        return f"I'm angry that \"{self.press_message}\" "


@dataclass
class ROF(_DAIDEObject):
    def __str__(self):
        return f"requesting an offer"


@dataclass
class ULB:
    power: Power
    float_val: float

    def __str__(self):
        return f"utility lower bound of float for {self.power} is {self.float_val} "


@dataclass
class UUB:
    power: Power
    float_val: float

    def __str__(self):
        return f"utility upper bound of float for {self.power} is {self.float_val} "


Reply = Union[YES, REJ, BWX, HUH, FCT, THK, IDK, WHY, POB, UHY, HPY, ANG]
PressMessage = Union[
    PRP, CCL, FCT, TRY, FRM, THK, INS, QRY, SUG, HOW, WHT, EXP, IFF, ULB, UUB
]
Message = Union[PressMessage, Reply]
Arrangement = Union[
    PCE,
    ALYVSS,
    DRW,
    XDO,
    DMZ,
    AND,
    ORR,
    SCD,
    CHO,
    FOR,
    XOY,
    YDO,
    SND,
    FWD,
    BCC,
    ROF,
]

AnyDAIDEToken = Union[
    RTO,
    DSB,
    BLD,
    REM,
    WVE,
    HLD,
    MTO,
    SUP,
    CVY,
    MoveByCVY,
    YES,
    REJ,
    BWX,
    HUH,
    WHY,
    POB,
    IDK,
    PRP,
    CCL,
    FCT,
    TRY,
    FRM,
    THK,
    INS,
    QRY,
    SUG,
    HOW,
    WHT,
    EXP,
    IFF,
    PCE,
    ALYVSS,
    DRW,
    XDO,
    DMZ,
    AND,
    ORR,
    SCD,
    CHO,
    FOR,
    XOY,
    YDO,
    SND,
    FWD,
    BCC,
    SLO,
    NOT,
    NAR,
    OCC,
    SRY,
    UHY,
    HPY,
    ANG,
    ROF,
    ULB,
    UUB,
]
