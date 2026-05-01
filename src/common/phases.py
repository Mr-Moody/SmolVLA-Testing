"""Single source of truth for the five-phase manipulation taxonomy.

All code that references phase IDs must import from here — never hardcode integers.
"""
from enum import IntEnum
from typing import Dict, Set


class Phase(IntEnum):
    FREE_MOTION = 0
    FINE_ALIGN = 1
    CONTACT_ESTABLISH = 2
    CONSTRAINED_MOTION = 3
    VERIFY_RELEASE = 4


PHASE_NAMES: Dict[Phase, str] = {
    Phase.FREE_MOTION: "free_motion",
    Phase.FINE_ALIGN: "fine_align",
    Phase.CONTACT_ESTABLISH: "contact_establish",
    Phase.CONSTRAINED_MOTION: "constrained_motion",
    Phase.VERIFY_RELEASE: "verify_release",
}

# Visual- and proprioceptive-cue grounded descriptions per task.
PHASE_DESCRIPTIONS: Dict[str, Dict[Phase, str]] = {
    "pick_place": {
        Phase.FREE_MOTION: (
            "Gripper moving toward the object or transporting object to target zone; "
            "no contact visible between gripper and object."
        ),
        Phase.FINE_ALIGN: (
            "Pre-grasp positioning; gripper visually servoing onto the object; "
            "TCP very close but not touching — no finger compression visible."
        ),
        Phase.CONTACT_ESTABLISH: (
            "Gripper fingers visibly compressing onto the object; "
            "gripper close command issued or force spike detected."
        ),
        Phase.CONSTRAINED_MOTION: (
            "Object grasped and being lifted or repositioned; "
            "gripper remains closed throughout."
        ),
        Phase.VERIFY_RELEASE: (
            "Object placed at target; gripper opening; "
            "arm retracting to clear zone."
        ),
    },
    "msd_plug": {
        Phase.FREE_MOTION: (
            "Gripper holding the connector, approaching the receptacle; "
            "no contact between connector pins and socket — connector in free air."
        ),
        Phase.FINE_ALIGN: (
            "Pin-to-socket sub-mm alignment; connector tips very close to socket face "
            "but not touching — no force spike yet."
        ),
        Phase.CONTACT_ESTABLISH: (
            "Pin tips touching the connector face / socket entry; "
            "first measurable Fz spike (pre-mate touch)."
        ),
        Phase.CONSTRAINED_MOTION: (
            "Insertion stroke in progress; connector sliding into socket under contact; "
            "Fz in nominal mate band."
        ),
        Phase.VERIFY_RELEASE: (
            "Connector fully seated; gripper opening; arm retracting; "
            "no contact alarm triggered."
        ),
    },
}

# Legal phase transitions.
# Rules:
#   - Forward-only progression (0→1→2→3→4).
#   - Self-loops allowed for all phases.
#   - FINE_ALIGN may fall back to FREE_MOTION (re-approach).
LEGAL_TRANSITIONS: Dict[Phase, Set[Phase]] = {
    Phase.FREE_MOTION: {Phase.FREE_MOTION, Phase.FINE_ALIGN},
    Phase.FINE_ALIGN: {Phase.FINE_ALIGN, Phase.CONTACT_ESTABLISH, Phase.FREE_MOTION},
    Phase.CONTACT_ESTABLISH: {Phase.CONTACT_ESTABLISH, Phase.CONSTRAINED_MOTION},
    Phase.CONSTRAINED_MOTION: {Phase.CONSTRAINED_MOTION, Phase.VERIFY_RELEASE},
    Phase.VERIFY_RELEASE: {Phase.VERIFY_RELEASE},
}


def is_legal_transition(prev: Phase, curr: Phase) -> bool:
    """Return True if transitioning from *prev* to *curr* is legal."""
    return curr in LEGAL_TRANSITIONS.get(prev, set())
