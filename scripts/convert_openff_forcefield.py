#!/usr/bin/env python3
"""
convert_openff_forcefield.py

Converts a real OpenFF SMIRNOFF force field file (an .offxml, e.g. Sage
"openff-2.1.0.offxml") into a compact JSON file js/openff-forcefield.js
loads directly in the browser -- no openff-toolkit, no OpenMM, pure
stdlib XML parsing. Deliberately NOT using the openff-toolkit Python
package to do this: the offxml format is plain, documented XML (SMIRNOFF
spec), and every value in a real Sage release file is a literal
`<number> * <unit>` string in one fixed unit per attribute (verified
directly against a real openff-2.1.0.offxml before writing this parser,
not assumed) -- so a small stdlib script can read it exactly, with no
extra runtime dependency this project's other converters don't already
avoid where possible.

What this does NOT convert, and why:
  - Constraints: irrelevant here -- this project only ever runs energy
    *minimization* (see js/openff-forcefield.js), never constrained MD,
    so a real Bonds harmonic term is used for every bond including X-H
    ones (checked directly: Sage's own Bonds section has real parameters
    covering H-containing bonds too, not just the constrained ones).
  - LibraryCharges / ToolkitAM1BCC: this project doesn't run AM1-BCC (no
    semiempirical QM in a browser). js/openff-forcefield.js uses this
    project's own NAGL-MBIS partial-charge model for electrostatics
    instead -- a real, documented, different-from-upstream substitution,
    not a hidden approximation (see OPENFF_INTEGRATION.md).
  - Fractional bond-order interpolation (k_bondorder1/2,
    length_bondorder1/2): Sage 2.1.0 defines zero Bonds or ProperTorsions
    parameters that use it (checked directly against the real file: 0/90
    bonds, 0/181 propertorsions) -- AM1-Wiberg fractional bond orders
    would need semiempirical QM this project doesn't have anyway, so this
    converter deliberately errors out if it ever finds one rather than
    silently dropping it (a future Sage release growing to use them would
    need real support added here, not silent wrong numbers).
  - Virtual sites: Sage 2.1.0's SMIRNOFF file has no VirtualSites section
    at all for this force field family; not handled.

Usage:
    python3 convert_openff_forcefield.py openff-2.1.0.offxml data/openff-sage-2.1.0.json
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

# Every attribute value in a real Sage offxml is "<number> * <unit...>"
# (or, rarely, just "<number>" with no unit at all, e.g. idivf). This
# regex pulls out just the leading number -- unit *strings* are checked
# against a fixed expectation per field below rather than parsed/converted
# generically, since a real Sage file uses exactly one unit per field
# (verified directly, see module docstring).
NUMBER_RE = re.compile(r'^\s*([-+0-9.eE]+)')


def parse_value(raw, expected_unit=None):
    m = NUMBER_RE.match(raw)
    if not m:
        raise ValueError('could not parse numeric value from %r' % raw)
    value = float(m.group(1))
    if expected_unit is not None:
        rest = raw[m.end():].strip()
        if rest.startswith('*'):
            rest = rest[1:].strip()
        if rest != expected_unit:
            raise ValueError('expected unit %r, got %r in %r' % (expected_unit, rest, raw))
    return value


DEG2RAD = 3.14159265358979323846 / 180.0


def convert_bonds(root):
    out = []
    for el in root.find('Bonds'):
        if 'k_bondorder1' in el.attrib:
            sys.exit('Bond %s uses fractional bond-order interpolation -- not supported, see module docstring' % el.attrib.get('id'))
        out.append({
            'smirks': el.attrib['smirks'],
            'id': el.attrib['id'],
            'length': parse_value(el.attrib['length'], 'angstrom'),
            'k': parse_value(el.attrib['k'], 'angstrom**-2 * mole**-1 * kilocalorie'),
        })
    return out


def convert_angles(root):
    out = []
    for el in root.find('Angles'):
        out.append({
            'smirks': el.attrib['smirks'],
            'id': el.attrib['id'],
            'angle': parse_value(el.attrib['angle'], 'degree') * DEG2RAD,
            'k': parse_value(el.attrib['k'], 'mole**-1 * radian**-2 * kilocalorie'),
        })
    return out


def convert_torsion_terms(el, prefix):
    terms = []
    n = 1
    while ('%s%d' % (prefix, n)) in el.attrib:
        if ('k%d_bondorder1' % n) in el.attrib:
            sys.exit('Torsion %s uses fractional bond-order interpolation -- not supported, see module docstring' % el.attrib.get('id'))
        k = parse_value(el.attrib['k%d' % n], 'mole**-1 * kilocalorie')
        idivf = float(el.attrib.get('idivf%d' % n, '1.0'))
        periodicity = int(el.attrib['periodicity%d' % n])
        phase = parse_value(el.attrib['phase%d' % n], 'degree') * DEG2RAD
        terms.append({'k': k / idivf, 'periodicity': periodicity, 'phase': phase})
        n += 1
    if not terms:
        sys.exit('Torsion %s has no k1/periodicity1/phase1 -- unexpected format' % el.attrib.get('id'))
    return terms


def convert_proper_torsions(root):
    out = []
    for el in root.find('ProperTorsions'):
        out.append({'smirks': el.attrib['smirks'], 'id': el.attrib['id'], 'terms': convert_torsion_terms(el, 'periodicity')})
    return out


def convert_improper_torsions(root):
    out = []
    for el in root.find('ImproperTorsions'):
        out.append({'smirks': el.attrib['smirks'], 'id': el.attrib['id'], 'terms': convert_torsion_terms(el, 'periodicity')})
    return out


SIGMA_TO_RMIN_HALF = 2.0 ** (1.0 / 6.0) / 2.0  # rmin = 2**(1/6) * sigma (standard 12-6 LJ relation)


def convert_vdw(root):
    vdw_el = root.find('vdW')
    out = []
    for el in vdw_el:
        if 'rmin_half' in el.attrib:
            rmin_half = parse_value(el.attrib['rmin_half'], 'angstrom')
        else:
            # The two TIP3P water entries (n-tip3p-O/H) are the only Sage
            # 2.1.0 vdW parameters expressed as sigma instead of rmin_half
            # (checked directly) -- converted via the standard sigma/rmin
            # relation rather than skipped, so a water-matching SMIRKS
            # still gets a real parameter if it's ever matched.
            rmin_half = parse_value(el.attrib['sigma'], 'angstrom') * SIGMA_TO_RMIN_HALF
        out.append({
            'smirks': el.attrib['smirks'],
            'id': el.attrib['id'],
            'epsilon': parse_value(el.attrib['epsilon'], 'mole**-1 * kilocalorie'),
            'rminHalf': rmin_half,
        })
    return out, {
        'scale14': parse_value(vdw_el.attrib['scale14']),
        'combiningRules': vdw_el.attrib.get('combining_rules'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('offxml', help='path to a SMIRNOFF .offxml file (e.g. openff-2.1.0.offxml)')
    parser.add_argument('output_json', help='path to write the converted JSON to')
    args = parser.parse_args()

    tree = ET.parse(args.offxml)
    root = tree.getroot()
    if root.tag != 'SMIRNOFF':
        sys.exit('%s does not look like a SMIRNOFF offxml (root tag is %r, expected SMIRNOFF)' % (args.offxml, root.tag))

    bonds_el = root.find('Bonds')
    angles_el = root.find('Angles')
    if bonds_el is None or angles_el is None:
        sys.exit('missing Bonds or Angles section -- not a usable small-molecule force field file')
    if bonds_el.attrib.get('potential') != 'harmonic' or angles_el.attrib.get('potential') != 'harmonic':
        sys.exit('unexpected Bonds/Angles potential (converter only implements harmonic)')

    torsion_potential = 'k*(1+cos(periodicity*theta-phase))'
    for sec in ('ProperTorsions', 'ImproperTorsions'):
        if root.find(sec).attrib.get('potential') != torsion_potential:
            sys.exit('unexpected %s potential (converter assumes %r)' % (sec, torsion_potential))

    vdw_list, vdw_meta = convert_vdw(root)
    elec_el = root.find('Electrostatics')

    data = {
        'source': args.offxml,
        'aromaticityModel': root.attrib.get('aromaticity_model'),
        'bonds': convert_bonds(root),
        'angles': convert_angles(root),
        'properTorsions': convert_proper_torsions(root),
        'improperTorsions': convert_improper_torsions(root),
        'vdw': vdw_list,
        'vdwScale14': vdw_meta['scale14'],
        'vdwCombiningRules': vdw_meta['combiningRules'],
        'electrostaticsScale14': parse_value(elec_el.attrib['scale14']) if elec_el is not None else 0.8333333333,
    }

    with open(args.output_json, 'w') as f:
        json.dump(data, f, indent=1)
        f.write('\n')

    print('Wrote %s: %d bonds, %d angles, %d proper torsions, %d improper torsions, %d vdW types' % (
        args.output_json, len(data['bonds']), len(data['angles']), len(data['properTorsions']),
        len(data['improperTorsions']), len(data['vdw'])))


if __name__ == '__main__':
    main()
