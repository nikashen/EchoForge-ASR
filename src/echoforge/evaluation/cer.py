from __future__ import annotations

from dataclasses import dataclass

from .normalize_zh import normalize_zh


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    hypothesis_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        if self.reference_units == 0:
            raise ValueError("CER is undefined for an empty normalized reference")
        return self.errors / self.reference_units


def edit_counts(reference: str, hypothesis: str, *, normalize: bool = True) -> EditCounts:
    ref = normalize_zh(reference) if normalize else reference
    hyp = normalize_zh(hypothesis) if normalize else hypothesis
    if not ref:
        raise ValueError("normalized reference must not be empty")

    # Each DP cell stores (total_cost, substitutions, deletions, insertions).
    previous = [(index, 0, index, 0) for index in range(len(ref) + 1)]
    for hyp_char in hyp:
        current = [(previous[0][0] + 1, 0, 0, previous[0][3] + 1)]
        for index, ref_char in enumerate(ref, start=1):
            if ref_char == hyp_char:
                diagonal = previous[index - 1]
            else:
                base = previous[index - 1]
                diagonal = (base[0] + 1, base[1] + 1, base[2], base[3])
            deletion_base = current[index - 1]
            deletion = (
                deletion_base[0] + 1,
                deletion_base[1],
                deletion_base[2] + 1,
                deletion_base[3],
            )
            insertion_base = previous[index]
            insertion = (
                insertion_base[0] + 1,
                insertion_base[1],
                insertion_base[2],
                insertion_base[3] + 1,
            )
            # Stable tie-break: match/substitution, then deletion, then insertion.
            current.append(min((diagonal, deletion, insertion), key=lambda item: item[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=len(ref),
        hypothesis_units=len(hyp),
    )


def character_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    return edit_counts(reference, hypothesis, normalize=normalize).cer
