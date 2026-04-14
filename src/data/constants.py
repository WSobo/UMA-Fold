"""
src/data/constants.py
──────────────────────
UMA-Fold data constants.

This is the subset of boltz.data.const that UMA-Fold actually uses, extracted
so the project no longer needs to import from a local boltz clone for constants.

IMPORTANT: The token vocabulary below must stay in sync with boltz's featurizer
output. The featurizer (still from boltz PyPI) maps residue types to integer IDs
using this exact vocabulary. If boltz changes its token list, update here too.
"""

# ── Token vocabulary ──────────────────────────────────────────────────────────
# 20 standard amino acids + UNK protein token
canonical_tokens = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",  # unknown protein token
]

# Full vocabulary: pad + gap + 21 protein + 5 RNA + 5 DNA = 33 tokens
tokens = [
    "<pad>",
    "-",
    *canonical_tokens,
    "A", "G", "C", "U", "N",       # RNA (N = unknown RNA)
    "DA", "DG", "DC", "DT", "DN",  # DNA (DN = unknown DNA)
]

token_ids: dict[str, int] = {token: i for i, token in enumerate(tokens)}
num_tokens: int = len(tokens)  # 33

unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}

# ── Contact conditioning ───────────────────────────────────────────────────────
# Used to determine s_input_dim in UMAFold:
#   s_input_dim = token_s + 2 * num_tokens + 1 + len(pocket_contact_info)
#                = 384    + 66              + 1 + 4  = 455
pocket_contact_info: dict[str, int] = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}
