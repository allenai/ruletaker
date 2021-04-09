from common import (
    Example,
    Fact,
    Rule,
    Theory,
    TheoryAssertionInstance,
    TheoryAssertionRepresentation,
    supported_theorem_provers,
)
import json
import pytest
from theory_label_generator import call_theorem_prover


def create_example_object():
    fact1 = Fact(polarity="+", predicate="green", arguments=["'Erin'"])
    fact2 = Fact(polarity="+", predicate="blue", arguments=["'Fiona'"])
    fact3 = Fact(polarity="+", predicate="big", arguments=["'Charlie'"])
    rules = [
        Rule(
            lhs=[Fact(polarity="+", predicate="green", arguments=["X"])],
            rhs=Fact(polarity="+", predicate="big", arguments=["X"]),
            prob=1.0,
        )
    ]
    theory = Theory(facts=[fact1, fact2, fact3], rules=rules)
    assertion = Fact(polarity="+", predicate="big", arguments=["'Erin'"])
    theory_assertion_instance = TheoryAssertionInstance(theory, assertion, True)
    logical_forms = TheoryAssertionRepresentation(
        theory_assertion_instance.theory.statements_as_texts,
        str(theory_assertion_instance.assertion),
    )
    fact_nls = [f.nl() for f in theory_assertion_instance.theory.facts]
    rule_nls = [r.nl() for r in theory_assertion_instance.theory.rules]
    assertion_nl = theory_assertion_instance.assertion.nl()
    english = TheoryAssertionRepresentation(fact_nls + rule_nls, assertion_nl)
    logic_program = dict()
    for theorem_prover in supported_theorem_provers:
        fact_lfs = []
        rule_lfs = []
        for fact in theory_assertion_instance.theory.facts:
            fact_lf = fact.logical_form(theorem_prover)
            fact_lfs.append(fact_lf)
        for rule in theory_assertion_instance.theory.rules:
            rule_lf = rule.logical_form(theorem_prover)
            rule_lfs.append(rule_lf)
        assertion_lf = theory_assertion_instance.assertion.logical_form(
            theorem_prover, is_assertion=True
        )
        logic_program[theorem_prover] = TheoryAssertionRepresentation(
            fact_lfs + rule_lfs, assertion_lf
        )
    return Example(
        str(1),
        theory_assertion_instance=theory_assertion_instance,
        logical_forms=logical_forms,
        english=english,
        logic_program=logic_program,
    )


# class RuletakerTestClass:

# def __init__(self):
example_obj = create_example_object()


def test_deserialization():
    json = example_obj.to_json()
    deserialized_obj = Example.from_json(json)
    assert example_obj == deserialized_obj


def test_label_generation_from_theorem_provers():
    theory_assertion_instance = example_obj.theory_assertion_instance
    for theorem_prover in supported_theorem_provers:
        label, _, _ = call_theorem_prover(
            theorem_prover,
            1,
            1,
            theory_assertion_instance.theory,
            theory_assertion_instance.assertion,
            theory_assertion_instance.label,
        )
        assert label == theory_assertion_instance.label
