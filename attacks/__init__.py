from attacks.ifgsm import IFGSM
from attacks.mifgsm import MIFGSM
from attacks.difgsm import DIFGSM
from attacks.tifgsm import TIFGSM
from attacks.sinifgsm import SINIFGSM
from attacks.freq_only import FreqOnlyAttack
from attacks.vit_aware import ViTAwareAttack
from attacks.sv_fca import SVFCAAttack

# Backward-compatible names. All legacy "ours"/SV-FTA/DDC entry points now
# resolve to the finalized SV-FCA implementation.
ATTACKS = {
    "ifgsm": IFGSM,
    "mifgsm": MIFGSM,
    "difgsm": DIFGSM,
    "tifgsm": TIFGSM,
    "si_ni_fgsm": SINIFGSM,
    "freq_only": FreqOnlyAttack,
    "vit_aware": ViTAwareAttack,
    "ours": SVFCAAttack,
    "sv_fca": SVFCAAttack,
    "svfca": SVFCAAttack,
    "sv_fta": SVFCAAttack,
    "svfta": SVFCAAttack,
    "ddc": SVFCAAttack,
}


def build_attack(name, **kwargs):
    if name not in ATTACKS:
        raise KeyError("Unknown attack '{}'. Available: {}".format(name, sorted(ATTACKS)))
    return ATTACKS[name](**kwargs)
