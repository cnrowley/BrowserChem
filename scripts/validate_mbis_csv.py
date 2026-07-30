import csv, re, ast, sys

def validate(path_or_text, is_text=False):
    reader = csv.DictReader(path_or_text.splitlines()) if is_text else csv.DictReader(open(path_or_text))
    n_rows = 0
    n_atom_mismatch = 0
    n_parse_fail = 0
    all_charges = []
    max_map_seen = []
    for row in reader:
        n_rows += 1
        smi = row['smiles']
        map_nums = sorted(int(n) for n in re.findall(r':(\d+)\]', smi))
        try:
            charges = ast.literal_eval(row['mbis_charge'])
        except Exception as e:
            n_parse_fail += 1
            print(f"row {n_rows}: FAILED TO PARSE charge list: {e}")
            continue

        if len(map_nums) != len(charges):
            n_atom_mismatch += 1
            print(f"row {n_rows}: MISMATCH -- {len(map_nums)} mapped atoms vs {len(charges)} charges")
            continue

        # map numbers should be a contiguous 1..N sequence with no gaps/dupes
        if map_nums != list(range(1, len(map_nums) + 1)):
            print(f"row {n_rows}: map numbers aren't a clean 1..N sequence: {map_nums[:10]}...")

        all_charges.extend(charges)
        max_map_seen.append(len(map_nums))

    print(f"\n{n_rows} rows checked")
    print(f"parse failures: {n_parse_fail}")
    print(f"atom-count/charge-count mismatches: {n_atom_mismatch}")
    if all_charges:
        print(f"charge value range: [{min(all_charges):.4f}, {max(all_charges):.4f}]")
        print(f"mean: {sum(all_charges)/len(all_charges):.4f}")
        # MBIS charges are typically small; flag anything wildly outside a
        # sane physical range as worth a second look, not proof of a bug.
        extreme = [c for c in all_charges if abs(c) > 2.0]
        print(f"values with |charge| > 2.0 (worth spot-checking, not necessarily wrong): {len(extreme)}")
    print(f"molecule sizes (atom count) seen: {sorted(set(max_map_seen))}")

sample = """smiles,mbis_charge
[H:23][C:10]12[C:8]3([N:7]([C:6]([N:12]1[H:25])([C:4]4([N:3]([C:2]([N:11]2[H:24])([C:1]([N:9]3[H:22])([N+:5]4([H:17])[H:18])[H:13])[H:14])[H:15])[H:16])[H:19])[H:20])[H:21],"[0.21417281764429827, 0.2937358958030937, -0.6647348257804334, 0.20650215522525373, -0.49291634891624597, 0.2404940703431581, -0.7274113957831827, 0.2650650953149744, -0.6401360929573164, 0.2806297114651914, -0.695567141306002, -0.6548006414075234, 0.12497940289408267, 0.11272234727008708, 0.38593385829258275, 0.11904211782161135, 0.3955721687379366, 0.405480778938144, 0.11181628542541244, 0.3885586859394625, 0.1148460373259739, 0.3709359240599796, 0.10691906243169035, 0.3722884549254082, 0.3659381652850469]"
[H:14][C:2]12[C:1]3([N:6]([C:5]4([C:4]([N:3]1[H:15])([O:9][C:8]([O:7]4)([C:10]([O:12]2)([O:11]3)[H:20])[H:19])[H:16])[H:17])[H:18])[H:13],"[0.31010781227495715, 0.2761394150504358, -0.7082136908433827, 0.2761394992922474, 0.3101078210593752, -0.6796720420674431, -0.39902339911092455, 0.3145324445990267, -0.42786705261758895, 0.3145324396950522, -0.3990234446158889, -0.42786705877348213, 0.08575208598878073, 0.08837157747057657, 0.3664743887552011, 0.08837155417181744, 0.08575206978599285, 0.352917241247323, 0.08616409938858374, 0.08616409704003308]"
[H:6][C:1]1([C:2]([C:3]1([Br:4])[Br:5])([H:8])[H:9])[H:7],"[-0.2960127545337287, -0.29601278100425754, 0.035013598584790626, -0.08354206651294564, -0.08354167603888228, 0.18103395205720663, 0.18103409473099893, 0.1810339429878796, 0.18103410270913617]"
[H:1][C:2]1=[N:3][O:4][C:5](=[N:14]1)[C:6]2([C:8]([C:11]2([Br:12])[Br:13])([H:9])[H:10])[H:7],"[0.12889955926103377, 0.3113793667878981, -0.2933244219253706, -0.14626064852291287, 0.6265869381468241, -0.2725126955654748, 0.21414187749281993, -0.2803179812557862, 0.1975744905191627, 0.18873836522381301, 0.02078946463419245, -0.046783626252431924, -0.06320372379661203, -0.5857254943520557]"
"""

validate(sample, is_text=True)

if __name__ == "__main__" and len(sys.argv) > 1:
    validate(sys.argv[1])
