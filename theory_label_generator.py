#!/usr/bin/env python
import argparse
from common import Example, Fact, Rule, Theory, TheoryAssertionRepresentationWithLabel
import json

import problog
from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
from problog.formula import LogicFormula, LogicDAG
from problog.sdd_formula import SDD
from problog.engine import NonGroundProbabilisticClause, UnknownClause
from problog.engine_stack import NegativeCycle

import re
import time

from utils import parse_statement

current_milli_time = lambda: int(round(time.time() * 1000))

ruletaker_variable_nl_to_variable_format = {"someone": "X", "something": "Y"}


class Metrics:
    """Class to store accuracy and timing related metrics when running an entire theories dataset
    through a theorem proving engine."""

    def __init__(self):
        self.num_examples = 0
        self.num_true = 0
        self.num_false = 0
        self.num_correct_true = 0
        self.num_correct_false = 0
        self.num_correct = 0
        self.total_elapsed_millisecs = 0
        self.num_true_with_exception = 0
        self.num_false_with_exception = 0
        self.num_correct_true_with_exception = 0
        self.num_correct_false_with_exception = 0
        self.num_incorrect_true_no_exception = 0
        self.num_incorrect_false_no_exception = 0
        self.num_no_gold_label = 0
        self.exception_num_failures = dict()

    def update(self, gold_label, engine_label, engine_exception, elapsed_millisecs):
        """Update metrics. To be called after processing each example from the dataset."""
        self.num_examples += 1
        self.total_elapsed_millisecs += elapsed_millisecs
        if gold_label is None:
            self.num_no_gold_label += 1
        else:
            engine_label_correct = gold_label == engine_label
            if not engine_label_correct:
                exception_msg = "No Exception"
                if engine_exception is not None:
                    exception_msg = engine_exception
                    if engine_exception not in self.exception_num_failures:
                        self.exception_num_failures[engine_exception] = 0
                    self.exception_num_failures[engine_exception] += 1
            if gold_label:
                self.num_true += 1
                if engine_label:
                    self.num_correct_true += 1
                if engine_exception is not None:
                    self.num_true_with_exception += 1
                    if engine_label:
                        self.num_correct_true_with_exception += 1
                else:
                    if not engine_label:
                        self.num_incorrect_true_no_exception += 1
            else:
                self.num_false += 1
                if not engine_label:
                    self.num_correct_false += 1
                if engine_exception is not None:
                    self.num_false_with_exception += 1
                    if not engine_label:
                        self.num_correct_false_with_exception += 1
                else:
                    if engine_label:
                        self.num_incorrect_false_no_exception += 1
            self.num_correct = self.num_correct_true + self.num_correct_false

    def report(self):
        """Report summarizing the overall accuracy, and breakdown by True and False (gold)
        labels. Also reports the number of examples that result in exceptions from the
        underlying engine, and timing information."""
        if self.num_examples > 0:
            avg_elapsed_secs = (self.total_elapsed_millisecs / self.num_examples) / 1000
            print(f"Total no. of examples: {self.num_examples}")
            if self.num_no_gold_label > 0:
                print(f"Found {self.num_no_gold_label} examples without a gold label")
            else:
                total_no_of_exceptions = (
                    self.num_true_with_exception + self.num_false_with_exception
                )
                print(f"  No. true: {self.num_true}")
                print(f"    No. correct: {self.num_correct_true}")
                print(f"    No. of exceptions: {self.num_true_with_exception}")
                print(
                    f"        No. correct with exceptions: {self.num_correct_true_with_exception}"
                )
                print(
                    f"    No. incorrect without exception: {self.num_incorrect_true_no_exception}"
                )
                print(f"  No. false: {self.num_false}")
                print(f"    No. correct: {self.num_correct_false}")
                print(f"    No. of exceptions: {self.num_false_with_exception}")
                print(
                    f"        No. correct with exceptions: {self.num_correct_false_with_exception}"
                )
                print(
                    f"    No. incorrect without exception: {self.num_incorrect_false_no_exception}"
                )
                print(f"Total no. correct: {self.num_correct}")
                print(f"Total no. with exceptions: {total_no_of_exceptions}")
                print(f"Accuracy: {(self.num_correct * 100.0) / self.num_examples}")
                if total_no_of_exceptions > 0:
                    print("\nFailure Breakdown by Exception:")
                    for exception in self.exception_num_failures:
                        print(
                            f"    {exception}: {self.exception_num_failures[exception]}"
                        )
            print(
                f"\nAverage theorem proving time per example: {avg_elapsed_secs} secs\n\n"
            )


def format_argument(arg_as_str):
    """Function that takes a string representing a predicate argument and formats it appropriately
    depending on whether it is a constatn or a variable.
    """
    arg_as_str = arg_as_str.lower()
    if arg_as_str in ruletaker_variable_nl_to_variable_format.keys():
        # If it's in the mapping, it is a variable, so return an appropriately formatted variable.
        return ruletaker_variable_nl_to_variable_format[arg_as_str]
    # If it's not in the mapping, it is a constant, so return a lower-cased string.
    return arg_as_str


def parse_triple_representation(triple_rep):
    """Function that takes string containing a triple representation in RuleTaker format and creates
    a Fact. E.g. input:
        (\"cow\" \"needs\" \"bear\" \"+\")
    """
    fact = None
    triple_rep = triple_rep.strip()
    # Remove enclosing parens ()
    triple_txt = triple_rep[1:-1]

    # Extract the parts of the triple by looking for quotes.
    # Replace spaces in predicate/args with underscores to make them valid terms.
    triple_parts = []
    for m in re.finditer(r'"([^"]+)"', triple_txt):
        triple_part = m.group(1).replace(" ", "_")
        triple_parts.append(triple_part)

    if len(triple_parts) == 4:
        arg1 = format_argument(triple_parts[0])
        predicate = triple_parts[1]
        arg2 = format_argument(triple_parts[2])
        polarity = triple_parts[3]
        if predicate == "is":
            predicate = f"{predicate}_{arg2}"
            fact = Fact(polarity, predicate, [arg1])
        else:
            fact = Fact(polarity, predicate, [arg1, arg2])
    return fact


def parse_rule_representation(rule_rep):
    """Function that takes string containing a rule in RuleTaker format and creates
    a Rule. E.g. input:
        (((\"something\" \"needs\" \"cow\" \"+\")) -> (\"something\" \"is\" \"red\" \"+\"))
    """
    rule = None
    rule_rep = rule_rep.strip()
    # Remove enclosing parens ()
    rule_txt = rule_rep[1:-1]
    rule_parts = rule_txt.split("->")
    if len(rule_parts) == 2:
        # LHS is enclosed in parens. Remove ().
        lhs = rule_parts[0].strip()[1:-1]
        rhs = rule_parts[1]
        lhs_facts = []
        lhs_parts = []
        for m in re.finditer(r"\([^()]+\)", lhs):
            lhs_part = m.group(0)
            lhs_fact = parse_triple_representation(lhs_part)
            if lhs_fact is not None:
                lhs_facts.append(lhs_fact)
        rhs_fact = parse_triple_representation(rhs)
        rule = Rule(lhs_facts, rhs_fact)
        return rule


def call_theorem_prover(
    theorem_prover, instance_id, question_id, theory, assertion, gold_label
):
    """Function that takes a single theory/assertion example and runs it through the theorem prover
    to obtain a label. Returns the obtained label, elapsed time to solve it, and exception returned
    by the engine, if any.
    """
    obtained_result = False
    millisecs_elapsed = 0
    print("=======ORIGINAL THEORY=========")
    theory_as_txt = theory.program(theorem_prover)
    print(theory_as_txt)
    theory.preprocess(theorem_prover)
    theory_as_txt = theory.program(theorem_prover)
    if theorem_prover == "problog":
        assertion_lf = assertion.logical_form(theorem_prover, False)
        assertion_lf = f"query({assertion_lf})."
        program = f"{theory_as_txt}\n{assertion_lf}"
        print("=======PROGRAM FROM PREPROCESSED THEORY=========")
        print(program)
        print("=======EXPECTED LABEL=========")
        print(f"    {gold_label}")
        start_millisecs = current_milli_time()
        try:
            lf = LogicFormula.create_from(program)  # ground the program
            dag = LogicDAG.create_from(lf)  # break cycles in the ground program
            sdd = SDD.create_from(dag)
            result = sdd.evaluate()
            end_millisecs = current_milli_time()
            elapsed_millisecs = end_millisecs - start_millisecs
            result_tuples = [(k, v) for k, v in result.items()]
            obtained_result = result_tuples[0][1] != 0.0
            return obtained_result, elapsed_millisecs, None
        except (NegativeCycle, NonGroundProbabilisticClause, UnknownClause) as e:
            end_millisecs = current_milli_time()
            elapsed_millisecs = end_millisecs - start_millisecs
            print(
                f"!!!Encountered Exception at instance id {instance_id}, question id {question_id}: {e}"
            )
            obtained_result = assertion.polarity != "+"
            exception_name = str(type(e)).lstrip("<class '").rstrip("'>")
            return obtained_result, elapsed_millisecs, exception_name
    return obtained_result, elapsed_millisecs, None


def run_theorem_prover(theorem_prover, ip, ip_format, op, report_metrics):
    """Function that takes an input file, calls the theorem prover on every example and gets a label.
    Results are written to output file. Metrics are tracked and reported if report_metrics is True.
    """
    metrics = Metrics()
    if ip_format == "current":
        row_ix = 1
        for ix, line in enumerate(ip.readlines()):
            facts = []
            rules = []
            instance_json = json.loads(line)
            instance = TheoryAssertionRepresentationWithLabel.from_json(instance_json)
            if instance is not None:
                for lf_str in instance.theory_statements:
                    statement = parse_statement(lf_str)
                    if isinstance(statement, Fact):
                        facts.append(statement)
                    elif isinstance(statement, Rule):
                        rules.append(statement)
                    else:
                        print(
                            f"Unable to parse statement {lf_str} in row {row_ix} of input jsonl file!"
                        )
                assertion = parse_statement(instance.assertion_statement)
                gold_label = instance.label
                theory = Theory(facts, rules)
                ix = str(row_ix)
                (
                    engine_label,
                    elapsed_millisecs,
                    returned_exception,
                ) = call_theorem_prover(
                    theorem_prover, ix, ix, theory, assertion, gold_label
                )
                if report_metrics:
                    metrics.update(
                        gold_label, engine_label, returned_exception, elapsed_millisecs
                    )
                instance_json["label"] = engine_label
                json.dump(instance_json, op)
                op.write("\n")
            else:
                print(f"Unexpected input file format in line no. {row_ix}")
            row_ix += 1
    else:
        # Ruletaker Legacy Jsonl Format
        for ix, line in enumerate(ip.readlines()):
            facts = []
            rules = []
            instance = json.loads(line)
            triples = instance["triples"]
            ip_rules = instance.get("rules", [])
            questions = instance["questions"]
            for triple_key in triples:
                triple_obj = triples[triple_key]
                triple_rep = triple_obj["representation"]
                fact = parse_triple_representation(triple_rep)
                if fact is not None:
                    facts.append(fact)
            for rule_key in ip_rules:
                rule_obj = ip_rules[rule_key]
                rule_rep = rule_obj["representation"]
                rule = parse_rule_representation(rule_rep)
                if rule is not None:
                    rules.append(rule)
            theory = Theory(facts, rules)
            for question_key in questions:
                question_obj = questions[question_key]
                question_rep = question_obj["representation"]
                assertion = parse_triple_representation(question_rep)
                gold_label = question_obj.get("answer", None)
                (
                    engine_label,
                    elapsed_millisecs,
                    returned_exception,
                ) = call_theorem_prover(
                    theorem_prover, ix, question_key, theory, assertion, gold_label
                )
                if report_metrics:
                    metrics.update(
                        gold_label, engine_label, returned_exception, elapsed_millisecs
                    )
                op_obj = {
                    **instance,
                    **({f"{theorem_prover}_label": engine_label}),
                }
                json.dump(op_obj, op)
                op.write("\n")
    if report_metrics:
        metrics.report()


def main():
    """Tool that takes a collection of theory-assertion examples and runs them through a theorem prover.
    Supported input format 1:  Jsonl format with json objects represented as per the
    `TheoryAssertionRepresentationWithLabel` class.
    Sample:
    {   "json_class": "TheoryAssertionRepresentation",
        "theory_statements": [
            "1.0::kind('Fiona').",
            "1.0::rough('Dave').",
            "1.0::smart('Dave').",
            "1.0::quiet('Charlie').",
            "1.0::kind('Dave').",
            "1.0::white('Erin').",
            "1.0::young(X) :- white(X).",
            "1.0::smart(X) :- big(X), green(X).",
            "1.0::kind(X) :- round(X), smart(X).",
            "1.0::kind(X) :- quiet(X), round(X).",
            "1.0::rough(X) :- round(X), red(X)."
            "1.0::kind(X) :- quiet(X).", "1.0::furry(X) :- quiet(X), big(X)."
        ],
        "assertion_statement": "query(1.0::young('Dave').)."
    }
    Supported input format 2: Ruletaker's legacy Jsonl format (for AI2's internal use with existing RuleTaker datasets)
    Sample (there are additional fields not relevant and not shown here):
    {   "id": "AttNoneg-D3-319", ...
        "triples":{
            "triple1":
                "text":"Bob is cold.",
                "representation":"(\"Bob\" \"is\" \"cold\" \"+\")"
            },
            "triple2": {
                "text":"Erin is nice.",
                "representation":"(\"Erin\" \"is\" \"nice\" \"+\")"
            },
            "triple3":{
                "text":"Gary is nice.",
                "representation":"(\"Gary\" \"is\" \"nice\" \"+\")"
            },
            "triple4":{
                "text":"Harry is blue.",
                "representation":"(\"Harry\" \"is\" \"blue\" \"+\")"
            }
        },
        "rules":{
            "rule1":{
                "text":"Blue people are furry.",
                "representation":"(((\"someone\" \"is\" \"blue\" \"+\")) -> (\"someone\" \"is\" \"furry\" \"+\"))"
            },
            "rule2":{
                "text":"Nice people are furry.",
                "representation":"(((\"someone\" \"is\" \"nice\" \"+\")) -> (\"someone\" \"is\" \"furry\" \"+\"))"
            },
            "rule3":{
                "text":"Blue, big people are nice.",
                "representation":"(((\"someone\" \"is\" \"blue\" \"+\") (\"someone\" \"is\" \"big\" \"+\"))
                                        -> (\"someone\" \"is\" \"nice\" \"+\"))"
                },
            "rule4":{
                "text":"If someone is cold then they are quiet.",
                "representation":"(((\"someone\" \"is\" \"cold\" \"+\"))
                                        -> (\"someone\" \"is\" \"quiet\" \"+\"))"},
            }
        },
        "questions":{
            "Q1":{
                "question":"Erin is nice.",
                "answer":true,
                ...
                "representation":"(\"Erin\" \"is\" \"nice\" \"+\")"
            },
            "Q2":{
                "question":"Gary is not nice.",
                "answer":false,
                ...
                "representation":"(\"Gary\" \"is\" \"nice\" \"-\")"
            },
            "Q3":{
                "question":"Gary is furry.",
                "answer":true,
                "representation":"(\"Gary\" \"is\" \"furry\" \"+\")"
            }
        }
    }
    Output jsonl format: Same as above with an additional field "problog_label": <true|false>.
    """
    parser = argparse.ArgumentParser(
        description="Tool to run theories through a theorem prover."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Input jsonl file in either the current format or the legacy RuleTaker Jsonl format",
    )
    parser.add_argument(
        "--input-format",
        choices=["current", "legacy"],
        default="current",
        help="Input file format",
    )
    parser.add_argument(
        "--theorem-prover",
        default="problog",
        help="Thorem proving engine to use. Only supported one right now is problog.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output file containing the theorem prover's output for each theory-assertion instance input. \
        Output format will be the same as input format, so this will be either a CSV or a jsonl file.",
    )
    parser.add_argument(
        "--report-metrics",
        action="store_true",
        help="Flag that will cause metrics (accuracy against gold labels) to be tracked and reported",
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as ip, open(args.output_file, "w") as op:
        run_theorem_prover(
            args.theorem_prover, ip, args.input_format, op, args.report_metrics
        )


if __name__ == "__main__":
    main()
