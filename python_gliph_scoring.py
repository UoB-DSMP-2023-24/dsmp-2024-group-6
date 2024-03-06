import random

def loadMotifs(file):
    motif_p = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            motif, counts, avgref, topref, ove, p = line.strip().split('\t')
            motif_p[motif] = float(p)
    return motif_p

def getExpansionP(cdr3_list, clone_annotation_lines):
    crg_count_per_clone = getCountPerCloneInConvergenceGroup(cdr3_list, clone_annotation_lines)
    crg_cdr3_count = len(cdr3_list.split(' '))
    equal_or_surpass_score_counts = 0
    for s in range(1000):
        subsample_list = randomSubsample(clone_annotation_lines, crg_cdr3_count)
        this_score = 0
        for i in range(len(subsample_list)):
            CDR3b, TRBV, TRBJ, CDR3a, TRAV, TRAJ, patient, counts = subsample_list[i].split('\t')
            if counts == "Counts":
                counts = 0
            if int(counts) > 1:
                this_score += 1
        this_score = this_score / crg_cdr3_count
        if this_score >= crg_count_per_clone:
            equal_or_surpass_score_counts += 1
    if equal_or_surpass_score_counts == 0:
        equal_or_surpass_score_counts = 1
    expansion_p = equal_or_surpass_score_counts / 1000
    return expansion_p

def getConvergeceGroupSizeP(unique_clone_count):
    counts2scores = {
        1: 0.954980692,
        2: 0.029106402,
        3: 0.006190808,
        4: 0.002347221,
        5: 0.001456437,
        6: 0.001212588,
        7: 0.000862264,
        8: 0.000829941,
        9: 0.000476289,
        10: 0.000313723,
        11: 0.000240521,
        12: 0.00015401,
        13: 7.70048E-05,
        14: 7.41528E-05,
        15: 9.50677E-05,
        16: 8.55609E-05,
        17: 9.50677E-05,
        18: 8.55609E-05,
        19: 0.000111229,
        20: 8.08075E-05,
        21: 9.60184E-05,
        22: 7.51035E-05,
        23: 8.46103E-05,
        24: 8.36596E-05,
        25: 5.22872E-05,
        26: 3.80271E-05,
        27: 2.56683E-05,
        28: 3.99284E-05,
        29: 3.99284E-05,
        30: 3.04217E-05,
        31: 2.37669E-05,
        32: 2.85203E-05,
        33: 3.70764E-05,
        34: 3.42244E-05,
        35: 3.04217E-05,
        36: 3.32737E-05,
        37: 3.04217E-05,
        38: 3.32737E-05,
        39: 3.99284E-05,
        40: 2.9471E-05,
        41: 3.04217E-05,
        42: 3.2323E-05,
        43: 2.6619E-05,
        44: 1.14081E-05,
        45: 1.42602E-05,
        46: 1.52108E-05,
        47: 1.52108E-05,
        48: 1.71122E-05,
        49: 2.18656E-05,
        50: 1.42602E-05
    }
    if unique_clone_count in counts2scores:
        return counts2scores[unique_clone_count]
    else:
        return 1E-05

def getDefaultBackgroundFreqs(file):
    freq_array = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name, freq = line.strip().split('\t')
            freq_array.append(float(freq))
    return freq_array

def getSubjectsInConvergenceGroup(cdr3_list, clone_annotation_lines):
    crg_subjects = set()
    clones = cdr3_list.split(' ')
    for clone in clones:
        for line in clone_annotation_lines:
            if line.startswith(clone):
                CDR3b, TRBV, TRBJ, CDR3a, TRAV, TRAJ, patient, counts = line.split('\t')
                crg_subjects.add(patient)
    crg_subject_array = list(crg_subjects)
    return crg_subject_array

def getCountPerCloneInConvergenceGroup(cdr3_list, clone_annotation_lines):
    total = 0
    clones = cdr3_list.split(' ')
    for clone in clones:
        for line in clone_annotation_lines:
            if line.startswith(clone):
                CDR3b, TRBV, TRBJ, CDR3a, TRAV, TRAJ, patient, counts = line.split('\t')
                if counts == "Counts":
                    counts = 0
                if int(counts) > 1:
                    total += 1
    count_per_clone = total / len(clones)
    return count_per_clone

def getUniqueClonesInConvergenceGroup(cdr3_list, clone_annotation_lines):
    crg_clones = set()
    clones = cdr3_list.split(' ')
    for clone in clones:
        for line in clone_annotation_lines:
            if line.startswith(clone):
                CDR3b, TRBV, TRBJ, CDR3a, TRAV, TRAJ, patient, counts = line.split('\t')
                crg_clones.add(patient + "_" + TRBV + "_" + TRBJ + "_" + CDR3b)
    crg_clone_array = list(crg_clones)
    return crg_clone_array

def getUniqueCloneVgenes(crg_clone_array):
    vgenes = []
    for clone in crg_clone_array:
        patient, TRBV, TRBJ, CDR3b = clone.split('_')
        vgenes.append(TRBV)
    return vgenes

def getUniqueCloneCDR3lens(crg_clone_array):
    cdr3lens = []
    for clone in crg_clone_array:
        patient, TRBV, TRBJ, CDR3b = clone.split('_')
        cdr3lens.append(len(CDR3b))
    return cdr3lens

def getBackgroundVgeneFreqs(clone_annotation_lines):
    unique_clones = getUniqueClones(clone_annotation_lines)
    vgene_freqs = []

def getUniqueClones(clone_annotation_lines):
    crg_clones = set()
    for line in clone_annotation_lines:
        CDR3b, TRBV, TRBJ, CDR3a, TRAV, TRAJ, patient, counts = line.split('\t')
        crg_clones.add(TRBV + "_" + TRBJ + "_" + CDR3b)
    crg_clone_array = list(crg_clones)
    return crg_clone_array

def loadFile(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def hla_probability(N, k, x, A, S):
    p = 1
    number_of_pass_cutoff_draws = 0
    for sim in range(S):
        people_left = N
        hla_of_interest_left = A
        number_of_hla_chosen = 0
        for picks in range(k):
            rand = random.randint(0, people_left - 1)
            if rand < hla_of_interest_left:
                number_of_hla_chosen += 1
                hla_of_interest_left -= 1
            people_left -= 1
        if number_of_hla_chosen >= x:
            number_of_pass_cutoff_draws += 1
    p = number_of_pass_cutoff_draws / S
    return number_of_pass_cutoff_draws, p

def calculate_enrichment_p(freq_array, test_data, sims):
    depth = len(test_data)
    test_score = get_simpson_index(test_data)
    unselected_score_distribution = []
    pass_scores = 0
    for s in range(sims):
        picks = []
        for d in range(depth):
            pick = biased_random_pick(freq_array)
            picks.append(pick)
        score = get_simpson_index(picks)
        unselected_score_distribution.append(score)
        if score >= test_score:
            pass_scores += 1
    p = pass_scores / sims
    if p == 0:
        p = 1 / sims
    return p

def get_simpson_index(picklist):
    pick_freqs = {}
    for pick in picklist:
        if pick in pick_freqs:
            pick_freqs[pick] += 1 / len(picklist)
        else:
            pick_freqs[pick] = 1 / len(picklist)
    score = 1
    for freq in pick_freqs.values():
        score *= freq
    return score

def biased_random_pick(array):
    pick = random.random()
    as_much = 0
    for i in range(len(array)):
        as_much += array[i]
        if pick <= as_much:
            return i
    return len(array) - 1

def count_hla_carriers(patient_array, patient_hla_hash, hla):
    count = 0
    for patient in patient_array:
        if patient + "_" + hla in patient_hla_hash:
            count += 1
    return count

def load_hla_hash(hla_file, hla_array, patient_array, patient_hla_hash):
    with open(hla_file, 'r') as f:
        lines = f.readlines()
    unique_hlas = set()
    patient_hlas = {}
    for line in lines:
        fields = line.strip().split('\t')
        patient_array.append(fields[0])
        for f in fields[1:]:
            hla = f.split(':')[0]
            unique_hlas.add(hla)
            patient_hla_hash[fields[0] + "_" + hla] = hla
    hla_array.extend(list(unique_hlas))

def getHammingDist(seq1, seq2):
    mismatch_columns = 0
    for c in range(len(seq1)):
        if seq1[c] != seq2[c]:
            mismatch_columns += 1
    return mismatch_columns

def randomSubsample(array, depth=None):
    id_array = list(range(len(array)))
    random.shuffle(id_array)
    if depth is None:
        depth = len(array)
    if depth > len(array):
        depth = len(array)
    random_subsample = [array[id] for id in id_array[:depth]]
    return random_subsample

def fisher_yates_shuffle(array):
    for i in range(len(array) - 1, 0, -1):
        j = random.randint(0, i)
        array[i], array[j] = array[j], array[i]

def addToHash(hash, newkey):
    if newkey in hash:
        hash[newkey] += 1
    else:
        hash[newkey] = 1

def GatherOptions():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--convergence_file", type=str, help="Convergence file")
    parser.add_argument("--clone_annotations", type=str, help="Clone annotations file")
    parser.add_argument("--hla_file", type=str, help="HLA file")
    parser.add_argument("--p_depth", type=int, default=10000, help="P depth")
    parser.add_argument("--motif_pval_file", type=str, help="Motif p-value file")
    args = parser.parse_args()
    return args.convergence_file, args.clone_annotations, args.hla_file, args.p_depth, args.motif_pval_file

convergence_file, clone_annotation_file, individual_hlas, pdepth, motif_pvalue_file = GatherOptions()

motif_p = loadMotifs(motif_pvalue_file)

convergence_groups = loadFile(convergence_file)
clone_annotation_lines = loadFile(clone_annotation_file)

hlas = []
patients = []
patient_hla = {}
load_hla_hash(individual_hlas, hlas, patients, patient_hla)

background_vgene_frequencies = getDefaultBackgroundFreqs("rootdir/db/tcrb-human.v-freq.txt")

if clone_annotation_file:
    pass

background_cdr3_frequencies = getDefaultBackgroundFreqs("rootdir/db/tcrb-human.cdr3len-freq.txt")

if clone_annotation_file:
    pass

for c in range(len(convergence_groups)):
    count, name, cdr3_list = convergence_groups[c].split('\t')
    if int(count) > 1:
        print(f"\n\nEvaluating {name} ({count} members: {cdr3_list})")
        crg_subject_array = getSubjectsInConvergenceGroup(cdr3_list, clone_annotation_lines)
        crg_unique_clones = getUniqueClonesInConvergenceGroup(cdr3_list, clone_annotation_lines)
        print(f"\t{len(crg_subject_array)} subjects and {len(crg_unique_clones)} clones")
        motifs_here = {}
        for m in range(len(motif_list)):
            for clone in crg_unique_clones:
                if motif_list[m] in clone:
                    if motif_list[m] in motifs_here:
                        motifs_here[motif_list[m]] += 1
                    else:
                        motifs_here[motif_list[m]] = 1
        motif_line = ""
        print("Motifs: ", end="")
        motif_keys = list(motifs_here.keys())
        for k in range(len(motif_keys)):
            motif_line += f" {motif_keys[k]}({motifs_here[motif_keys[k]]}, {motif_p[motif_keys[k]]})"
        print(f"Motifs: {motif_line}")
        crg_Vbs = getUniqueCloneVgenes(crg_unique_clones)
        Vb_p = calculate_enrichment_p(background_vgene_frequencies, crg_Vbs, pdepth)
        crg_cdr3blens = getUniqueCloneCDR3lens(crg_unique_clones)
        cdr3blen_p = calculate_enrichment_p(background_cdr3_frequencies, crg_cdr3blens, pdepth)
        lowest_hla_score = 1
        lowest_hla = ""
        for h in range(len(hlas)):
            all_patient_count = len(patients)
            all_patient_hla_count = count_hla_carriers(patients, patient_hla, hlas[h])
            crg_patient_count = len(crg_subject_array)
            crg_patient_hla_count = count_hla_carriers(crg_subject_array, patient_hla, hlas[h])
            if crg_patient_hla_count > 1:
                number_of_pass_cutoff_draws, p = hla_probability(all_patient_count, crg_patient_count, crg_patient_hla_count, all_patient_hla_count, 100000)
                print(f"\t{hlas[h]}\t({crg_patient_hla_count}/{crg_patient_count}) vs ({all_patient_hla_count}/{all_patient_count})\t{number_of_pass_cutoff_draws}\t{p}")
                if p < lowest_hla_score:
                    lowest_hla_score = p
                if p < 0.1:
                    lowest_hla += f"{hlas[h]}({p}) "
        size_p = getConvergeceGroupSizeP(len(cdr3_list.split(' ')))
        expansion_p = getExpansionP(cdr3_list, clone_annotation_lines)
        motif_p = 0.001
        clones = cdr3_list.split(' ')
        for x in range(len(clones)):
            for z in range(len(clone_annotation_lines)):
                if clone_annotation_lines[z].startswith(clones[x]):
                    print(f"  {clone_annotation_lines[z]}")
        print("\nFinal scores:")
        print(f"\tVsegment_p\t{Vb_p}")
        print(f"\tcdr3len_p\t{cdr3blen_p}")
        print(f"\tlowest_hla\t{lowest_hla_score}\t{lowest_hla}")
        print(f"\texpansion\t{expansion_p}")
        print(f"\tcluster size\t{size_p}")
        print(f"\tmotifs\t{motif_p}")
        score = Vb_p * cdr3blen_p * lowest_hla_score * expansion_p * motif_p * size_p * 64
        print(f"\tFINAL SCORE = {score}")
        unique_subjects = len(crg_subject_array)
        unique_clones = len(crg_unique_clones)
        print("Name\tCDR3s\tSubjects\tClones\tCRG_Score\tVb_p\tCDR3_p\tHLA_p\tExpansion_p\tMotif_p\tSize_p\tHLA\tMotifs")
        print(f"{name}\t{count}\t{unique_subjects}\t{unique_clones}\t{score}", end="")
        print(f"\t{Vb_p}\t{cdr3blen_p}\t{lowest_hla_score}\t{expansion_p}\t{motif_p}\t{size_p}\t{lowest_hla}\t{motif_line}")

