from studio.stages.term_check import (
    TermFix,
    apply_term_fixes,
    find_suspect_terms,
    spelling_variants,
    verify_terms,
)


class FakeVerifier:
    def __init__(self, values: dict[str, int]) -> None:
        self.values = {key.lower(): value for key, value in values.items()}

    def hits(self, term: str, context: str = "") -> int | None:
        return self.values.get(term.lower(), 0)


def test_variants_and_verified_fix_preserve_inflections() -> None:
    assert "Курокрад" in spelling_variants("Курократ")
    terms = find_suspect_terms("Позвали Курократа. Спросили Курократа. Вот Курократ.")
    fixes = verify_terms(terms, FakeVerifier({"Курокрад": 9}))
    assert [(fix.wrong, fix.right) for fix in fixes] == [("Курократ", "Курокрад")]
    fixed, count = apply_term_fixes("Курократ и курократа", fixes)
    assert (fixed, count) == ("Курокрад и курокрада", 2)


def test_known_source_term_is_never_rewritten() -> None:
    terms = find_suspect_terms("Это Зеленоград. Дошли до Зеленограда.")
    assert verify_terms(terms, FakeVerifier({"Зеленоград": 500, "Зеленокрад": 900})) == []


def test_manual_fix_is_supported_without_network() -> None:
    assert apply_term_fixes("Курократа", [TermFix("Курократ", "Курокрад", 1, "")]) == (
        "Курокрада",
        1,
    )


def test_a_capital_the_fix_adds_on_purpose_is_kept() -> None:
    fix = TermFix("стекловерфлею", "Stack Overflow", 1, "")
    assert apply_term_fixes("со стекловерфлею, то", [fix]) == ("со Stack Overflow, то", 1)
    assert apply_term_fixes("Стекловерфлею!", [fix]) == ("Stack Overflow!", 1)
