"""Compatibility shim.

The project previously stored experimental DDC/SV-FTA code in this module.
The finalized method is SV-FCA and lives in attacks/sv_fca.py.
"""
from attacks.sv_fca import SVFCAAttack

DDCAttack = SVFCAAttack
SVFTAttack = SVFCAAttack
SVFTAAttack = SVFCAAttack
