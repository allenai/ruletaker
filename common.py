#!/usr/bin/env python
import copy
import random

supported_theorem_provers = ["problog"]
default_theorem_prover = "problog"

variables = ["X", "Y"]

# In the original RuleTaker dataset, variables were represented using the following NL.
# These are used to check if an argument is a variable when we are loading/processing
# existing RuleTaker theories.
variables_ruletaker = ["someone", "something"]

all_variables = variables + variables_ruletaker


class Fact:
    """Class to represent a simple fact in a theory. It basically consists of a predicate
    with its arguments, a polarity (positive/negative) for it, along with an associated
    probability."""

    def __init__(self, polarity, predicate, arguments, prob=1.0):
        self.polarity = polarity
        self.predicate = predicate
        self.arguments = arguments
        self.probability = prob

    def __repr__(self):
        return f'{self.polarity} ( {self.predicate} {", ".join(self.arguments)} )'

    def __eq__(self, other):
        return (
            isinstance(other, Fact)
            and self.polarity == other.polarity
            and self.predicate == other.predicate
            and self.arguments == other.arguments
            and self.probability == other.probability
        )

    def __lt__(self, other):
        return isinstance(other, Fact) and (
            repr(self) < repr(other)
            or (repr(self) == repr(other) and (self.probability < other.probability))
        )

    def __hash__(self):
        return hash(
            (self.polarity, self.predicate, tuple(self.arguments), self.probability)
        )

    @classmethod
    def from_json(cls, json_dict):
        json_class = json_dict.get("json_class")
        if json_class == "Fact":
            arguments = [argument for argument in json_dict["arguments"]]
            return Fact(
                json_dict["polarity"],
                json_dict["predicate"],
                arguments,
                json_dict.get("probability", 1.0),
            )
        return None

    def to_json(self):
        return {
            "json_class": "Fact",
            "polarity": self.polarity,
            "predicate": self.predicate,
            "arguments": self.arguments,
            "probability": self.probability,
        }

    def constants(self):
        return set([argument for argument in self.arguments if argument.islower()])

    def logical_form(self, theorem_prover, standalone=True, is_assertion=False):
        """Produce a logical form representation of the fact in specified theorem prover format."""
        lf = ""
        arguments = self.arguments
        if theorem_prover.lower() == "problog":
            prob = f"{self.probability}::"
            if self.polarity != "+":
                lf += "\+"
            lf += f'{self.predicate}({", ".join(arguments)})'
            if is_assertion:
                lf = f"query({lf})."
            elif standalone:
                lf = f"{prob}{lf}."
        return lf

    def nl(self, standalone=True):
        """Produce a simple English representation of the fact.
        If no. of arguments is 1, the predicate is assumed to be an attribute and NL will take the form:
        <Argument> is <Attribute>. If no. of arguments is 2, the predicate is assumed to be a relation where
        the first argument is the subject and second argument is the object, and NL will take the form:
        <Argument1> <Relation> <Argument2>. Additionally, if the polarity is not positive, then the phrase
        'It is not true that ' is prepended to the NL."""

        def format_argument_as_nl(arg):
            """Formats a predicates arguments for NL generation."""
            # A constant is in lowercase with open and close single quotes. The quotes need to be removed.
            # A variable appears as an uppercase letter, one of the letters in the variables collection
            # define above, and in this case we return the letter as is.
            return arg.lstrip("'").rstrip("'").replace("_", " ")

        fact_nl = ""
        if len(self.arguments) == 1:
            arg = format_argument_as_nl(self.arguments[0])
            fact_nl = f"{arg} is {self.predicate}"
        elif len(self.arguments) == 2:
            arg1 = format_argument_as_nl(self.arguments[0])
            arg2 = format_argument_as_nl(self.arguments[1])
            fact_nl = f"{arg1} {self.predicate} {arg2}"
        if len(fact_nl) > 0:
            negate = not ((self.polarity == "+") != (self.probability == float(0)))
            if negate:
                fact_nl = f"it is not true that {fact_nl}"
            if standalone:
                fact_nl += "."
        if standalone:
            fact_nl = fact_nl[0].upper() + fact_nl[1:]
        return fact_nl


class Rule:
    """Class to represent a rule in a theory, i.e., something of the form "If A then B". Antecendent
    (LHS) here is a collection of Facts and the Consequent (RHS) is a single Fact. The rule also
    has an associated probability."""

    def __init__(self, lhs, rhs, prob=1.0):
        self.lhs = lhs
        self.rhs = rhs
        self.probability = prob

    def __repr__(self):
        lhs_repr = f'({" ".join(str(lhs_part) for lhs_part in self.lhs)})'
        return f"{lhs_repr} -> {str(self.rhs)}"

    def __eq__(self, other):
        return (
            isinstance(other, Rule)
            and set(self.lhs) == set(other.lhs)
            and self.rhs == other.rhs
            and self.probability == other.probability
        )

    def sorted_lhs(self):
        lhs_reprs = [repr(fact) for fact in self.lhs]
        return " || ".join(sorted(lhs_reprs))

    def __lt__(self, other):
        return isinstance(other, Rule) and (
            self.sorted_lhs() < other.sorted_lhs()
            or (
                self.sorted_lhs() == other.sorted_lhs()
                and repr(self.rhs) < repr(other.rhs)
            )
            or (
                self.sorted_lhs() == other.sorted_lhs
                and self.rhs == other.rhs
                and self.probability < other.probability
            )
        )

    def __hash__(self):
        return hash((tuple(sorted(self.lhs)), self.rhs, self.probability))

    @classmethod
    def from_json(cls, json_dict):
        json_class = json_dict.get("json_class")
        if json_class == "Rule":
            lhs_facts = [Fact.from_json(fact) for fact in json_dict["lhs"]]
            return Rule(
                lhs_facts, Fact.from_json(json_dict["rhs"]), json_dict["probability"]
            )
        return None

    def to_json(self):
        lhs_facts = [fact.to_json() for fact in self.lhs]
        return {
            "json_class": "Rule",
            "lhs": lhs_facts,
            "rhs": self.rhs.to_json(),
            "probability": self.probability,
        }

    def constants(self):
        facts = self.lhs + [self.rhs]
        constants_in_rule = set()
        for fact in facts:
            constants_in_rule = constants_in_rule.union(fact.constants())
        return constants_in_rule

    def logical_form(self, theorem_prover, is_assertion=False):
        """Produce a logical form representation of the rule in specified theorem prover format."""
        lf = ""
        if theorem_prover.lower() == "problog":
            prob = f"{self.probability}::"
            antecedant_lf = ", ".join(
                [lhs_fact.logical_form(theorem_prover, False) for lhs_fact in self.lhs]
            )
            consequent_lf = self.rhs.logical_form(theorem_prover, False)
            lf = f"{prob}{consequent_lf} :- {antecedant_lf}."
            if is_assertion:
                lf = f"query({lf})."
        return lf

    def nl(self):
        """Produce a simple English representation of the rule.
        The LHS Facts are each converted to NL and joined together with 'and's in the middle.
        NL is generated for the RHS Fact. Then the two are joined together with the template
        If <LHS> then <RHS>.
        """
        lhs_nl_statements = [f.nl(standalone=False) for f in self.lhs]
        lhs_nl = " and ".join(lhs_nl_statements)
        rhs_nl = self.rhs.nl(standalone=False)
        if self.probability != float(0):
            nl = f"If {lhs_nl} then {rhs_nl}."
        else:
            nl = f"If {lhs_nl} then it is not true that {rhs_nl}."
        return nl


class Theory:
    """A "theory" is a collection of facts and rules."""

    def __init__(self, facts, rules, statements_as_texts=None):
        self.facts = facts
        self.rules = rules
        if statements_as_texts is None:
            self.statements_as_texts = []
            for fact in facts:
                self.statements_as_texts.append(str(fact))
            for rule in rules:
                self.statements_as_texts.append(str(rule))
        else:
            self.statements_as_texts = statements_as_texts

    def __eq__(self, other):
        return (
            isinstance(other, Theory)
            and set(self.facts) == set(other.facts)
            and set(self.rules) == set(other.rules)
        )

    def __hash__(self):
        return hash((tuple(sorted(self.facts)), tuple(sorted(self.rules))))

    @classmethod
    def from_json(cls, json_dict):
        json_class = json_dict.get("json_class")
        if json_class == "Theory":
            facts = [Fact.from_json(fact) for fact in json_dict["facts"]]
            rules = [Rule.from_json(rule) for rule in json_dict["rules"]]
            return Theory(facts, rules)
        return None

    def to_json(self):
        facts = [fact.to_json() for fact in self.facts]
        rules = [rule.to_json() for rule in self.rules]
        return {"json_class": "Theory", "facts": facts, "rules": rules}

    def constants(self):
        """All the constant terms that appear in this theory. Correspond to
        terminals in the grammar from which the theory was built."""
        constants_in_theory = set()
        for fact in self.facts:
            constants_in_theory = constants_in_theory.union(fact.constants())
        for rule in self.rules:
            constants_in_theory = constants_in_theory.union(rule.constants())
        return constants_in_theory

    def program(self, theorem_prover, assertion=None):
        """Creates a program for the theory in format expected by the theorem_prover."""
        fact_lfs = []
        rule_lfs = []
        for fact in self.facts:
            fact_lf = fact.logical_form(theorem_prover)
            fact_lfs.append(fact_lf)
        for rule in self.rules:
            rule_lf = rule.logical_form(theorem_prover)
            rule_lfs.append(rule_lf)
        prog = "\n".join(fact_lfs + rule_lfs)
        if theorem_prover == "problog" and assertion is not None:
            assertion_lf = assertion.logical_form(
                theorem_prover, standalone=False, is_assertion=True
            )
            prog += f"\n{assertion_lf}"
        return prog

    def nl(self):
        fact_nls = [f.nl() for f in self.facts]
        rule_nls = [r.nl() for r in self.rules]
        nl = " ".join(fact_nls + rule_nls)
        return nl

    def handle_unknown_clauses(self):
        """Preprocess theory to avoid UnknownClause errors arising from rule antecedants containing
        clauses (Facts) that are not defined in the theory (a problem that arises with Problog).
        This is done by adding dummy facts for the missing clauses."""

        def create_fact(predicate, arguments_in_theory, num_arguments, polarity):
            constants = arguments_in_theory - set(all_variables)
            args_to_choose_from = set(constants)
            num_missing_constants = num_arguments - len(constants)

            if num_missing_constants > 0:
                sampled_vars = random.sample(variables, num_missing_constants)
                args_to_choose_from.update(set(sampled_vars))

            arguments = random.sample(args_to_choose_from, num_arguments)
            fact = Fact("+", predicate, arguments, 0.0)
            return fact

        predicates_in_rule_antecedants = dict()
        predicates_in_rule_consequents = set()
        predicates_in_facts = set()
        arguments_in_theory = set()
        for fact in self.facts:
            if fact.polarity == "+":
                predicates_in_facts.add(fact.predicate)
            for arg in fact.arguments:
                arguments_in_theory.add(arg)
        for rule in self.rules:
            if rule.rhs.polarity == "+":
                predicates_in_rule_consequents.add(rule.rhs.predicate)

        new_facts = []
        for rule in self.rules:
            for lhs_fact in rule.lhs:
                predicates_in_rule_antecedants[lhs_fact.predicate] = (
                    lhs_fact.polarity,
                    len(lhs_fact.arguments),
                )
        rule_antecedant_predicates_not_in_facts = (
            predicates_in_rule_antecedants.keys()
            - (predicates_in_facts.union(predicates_in_rule_consequents))
        )
        for rule_antecedant_predicate in rule_antecedant_predicates_not_in_facts:
            polarity = predicates_in_rule_antecedants[rule_antecedant_predicate][0]
            num_args = predicates_in_rule_antecedants[rule_antecedant_predicate][1]
            new_fact = create_fact(
                rule_antecedant_predicate, arguments_in_theory, num_args, polarity
            )
            new_facts.append(new_fact)
        self.facts.extend(new_facts)

    def ground_rule(self, rule):
        """Helper that grounds variables in a given rule. Used to preprocess theories to make them
        Problog-friendly."""

        def ground_variable(rule, variable, constant):
            rule_copy = copy.deepcopy(rule)
            for fact in rule_copy.lhs:
                for i in range(len(fact.arguments)):
                    if fact.arguments[i] == variable:
                        fact.arguments[i] = constant

            for i in range(len(rule_copy.rhs.arguments)):
                if rule_copy.rhs.arguments[i] == variable:
                    rule_copy.rhs.arguments[i] = constant
            return rule_copy

        constants_in_theory = self.constants()
        variables = set()
        for fact in rule.lhs:
            for argument in fact.arguments:
                if argument in all_variables:
                    variables.add(argument)
        for argument in rule.rhs.arguments:
            if argument in all_variables:
                variables.add(argument)
        rules = [rule]
        new_rules = []
        for variable in variables:
            for rule in rules:
                for constant in constants_in_theory:
                    new_rule = ground_variable(rule, variable, constant)
                    new_rules.append(new_rule)
            rules = new_rules
        return rules

    def ground_variables_in_negated_rule_clauses(self):
        """Preprocess theory to ground variables in rules with negated clauses in antecedent.
        Again meant to make input theory friendly for Problog."""

        def has_variable_argument(fact):
            for argument in fact.arguments:
                if argument in all_variables:
                    return True
            return False

        def has_negated_antecedent_with_variable(rule):
            found_negated_antecedent_with_variable = False
            for fact in rule.lhs:
                if fact.polarity != "+" and has_variable_argument(fact):
                    found_negated_antecedent_with_variable = True
                    break
            return found_negated_antecedent_with_variable

        modified_rules = []
        for rule in self.rules:
            if has_negated_antecedent_with_variable(rule):
                grounded_rules = self.ground_rule(rule)
                modified_rules.extend(grounded_rules)
            else:
                modified_rules.append(rule)
        self.rules = modified_rules

    def preprocess(self, theorem_prover):
        """Preprocess theory to make it friendly to the theorem prover being used. Certain
        features of a theory may cause the engine to throw exceptions. Currently, this happens
        under several conditions, with Problog."""
        if theorem_prover == "problog":
            self.handle_unknown_clauses()
            self.ground_variables_in_negated_rule_clauses()


class TheoryAssertionInstance:
    """Class representing a theory-assertion pair instance to be input to a model.
    Consists a gold truth label for the assertion's truthiness with respect to the theory.
    The `exception` field is a placeholder to store any exceptions thrown by the theorem prover
    on existing theory datasets generated outside of ruletaker. Other theory datasets can be validated
    or evaluated by running them through theorem provers supported in this repo by using the
    `theory_label_generator` tool.
    `min_proof_depth` is an integer field containing the depth of the
    proof; the depth of the simplest (shortest) proof if there are multiple.
    `proof` is a string representation of the proof from the theorem prover.
    The proof related fields are only present (not None) if the `label` is True."""

    def __init__(
        self,
        theory,
        assertion,
        label=None,
        exception=None,
        min_proof_depth=None,
        proof=None,
    ):
        self.theory = theory
        self.assertion = assertion
        self.label = label
        self.exception = exception
        self.min_proof_depth = min_proof_depth
        self.proof = proof

    def __eq__(self, other):
        return (
            isinstance(other, TheoryAssertionInstance)
            and self.theory == other.theory
            and self.assertion == other.assertion
            and self.label == other.label
            and self.exception == other.exception
            and self.min_proof_depth == other.min_proof_depth
            and self.proof == other.proof
        )

    def __hash__(self):
        return hash(
            (
                self.theory,
                self.assertion,
                self.label,
                self.exception,
                self.min_proof_depth,
                self.proof,
            )
        )

    @classmethod
    def from_json(cls, json_dict):
        json_class = json_dict.get("json_class")
        if json_class == "TheoryAssertionInstance":
            return TheoryAssertionInstance(
                Theory.from_json(json_dict["theory"]),
                Fact.from_json(json_dict["assertion"]),
                json_dict.get("label"),
                json_dict.get("exception"),
                json_dict.get("min_proof_depth"),
                json_dict.get("proof"),
            )
        return None

    def to_json(self):
        return {
            "json_class": "TheoryAssertionInstance",
            "theory": self.theory.to_json(),
            "assertion": self.assertion.to_json(),
            "label": self.label,
            "exception": self.exception,
            "min_proof_depth": self.min_proof_depth,
            "proof": self.proof,
        }


class TheoryAssertionRepresentation:
    """Class to encapsulate different representations for a TheoryAssertionInstance in an Example.
    The representations provided currently are the logical forms (prefix notation), natural language,
    and logic program in theorem prover format. This class consists of representations in one of the
    aforementioned forms for the facts and rules in a theory,and the corresponding representation for
    the assertion."""

    def __init__(self, theory_statements, assertion_statement):
        # Collection of strings
        self.theory_statements = theory_statements
        # String
        self.assertion_statement = assertion_statement

    def __hash__(self):
        return hash((tuple(self.theory_statements), self.assertion_statement))

    def __eq__(self, other):
        return (
            isinstance(other, TheoryAssertionRepresentation)
            and set(self.theory_statements) == set(other.theory_statements)
            and self.assertion_statement == other.assertion_statement
        )

    @classmethod
    def from_json(cls, json_dict):
        json_class = json_dict.get("json_class")
        if json_class == "TheoryAssertionRepresentation":
            return TheoryAssertionRepresentation(
                json_dict["theory_statements"], json_dict["assertion_statement"]
            )
        return None

    def to_json(self):
        return {
            "json_class": "TheoryAssertionRepresentation",
            "theory_statements": self.theory_statements,
            "assertion_statement": self.assertion_statement,
        }


class Example:
    """Class representing a generated example, which constitutes a TheoryAssertionInstance
    and its representations as logical forms in prefix notation, natural language, and
    logic programs in theorem prover formats."""

    def __init__(
        self,
        id,
        theory_assertion_instance,
        logical_forms=None,
        english=None,
        logic_program=None,
    ):
        self.id = id
        self.theory_assertion_instance = theory_assertion_instance
        if logical_forms is not None:
            self.logical_forms = logical_forms
        else:
            self.logical_forms = TheoryAssertionRepresentation(
                self.theory_assertion_instance.theory.statements_as_texts,
                str(self.theory_assertion_instance.assertion),
            )
        if english is not None:
            self.english = english
        else:
            fact_nls = [f.nl() for f in self.theory_assertion_instance.theory.facts]
            rule_nls = [r.nl() for r in self.theory_assertion_instance.theory.rules]
            assertion_nl = self.theory_assertion_instance.assertion.nl()
            self.english = TheoryAssertionRepresentation(
                fact_nls + rule_nls, assertion_nl
            )
        if logic_program is not None:
            self.logic_program = logic_program
        else:
            self.logic_program = dict()
            for theorem_prover in supported_theorem_provers:
                fact_lfs = []
                rule_lfs = []
                for fact in self.theory_assertion_instance.theory.facts:
                    fact_lf = fact.logical_form(theorem_prover)
                    fact_lfs.append(fact_lf)
                for rule in self.theory_assertion_instance.theory.rules:
                    rule_lf = rule.logical_form(theorem_prover)
                    rule_lfs.append(rule_lf)
                assertion_lf = self.theory_assertion_instance.assertion.logical_form(
                    theorem_prover, is_assertion=True
                )
                self.logic_program[theorem_prover] = TheoryAssertionRepresentation(
                    fact_lfs + rule_lfs, assertion_lf
                )

    def __eq__(self, other):
        return (
            isinstance(other, Example)
            and self.id == other.id
            and self.theory_assertion_instance == other.theory_assertion_instance
            and self.logical_forms == other.logical_forms
            and self.english == other.english
            and self.logic_program == other.logic_program
        )

    def __hash__(self):
        return hash(
            (
                self.id,
                self.theory_assertion_instance,
                self.logical_forms,
                self.english,
                self.logic_program,
            )
        )

    @classmethod
    def from_json(cls, json_dict):
        json_class = json_dict.get("json_class")
        if json_class == "Example":
            logic_program = dict()
            for k in json_dict["logic_program"]:
                logic_program[k] = TheoryAssertionRepresentation.from_json(
                    json_dict["logic_program"][k]
                )
            return Example(
                json_dict["id"],
                TheoryAssertionInstance.from_json(
                    json_dict["theory_assertion_instance"]
                ),
                TheoryAssertionRepresentation.from_json(json_dict.get("logical_forms")),
                TheoryAssertionRepresentation.from_json(json_dict.get("english")),
                logic_program,
            )
        return None

    def to_json(self):
        logic_program = dict()
        for k in self.logic_program:
            logic_program[k] = self.logic_program[k].to_json()
        return {
            "json_class": "Example",
            "id": self.id,
            "theory_assertion_instance": self.theory_assertion_instance.to_json(),
            "logical_forms": self.logical_forms.to_json(),
            "english": self.english.to_json(),
            "logic_program": logic_program,
        }


class TheoryAssertionRepresentationWithLabel:
    """Class that represents the structure of expected input to theory_label_generator. Contains
    theory statements, which is a collection of strings, a string representing the assertion. When
    input to theory_label_generator these statements would be logical forms in prefix notation."""

    def __init__(self, id, theory_statements, assertion_statement, label=None):
        # String
        self.id = id
        # Collection of strings
        self.theory_statements = theory_statements
        # String
        self.assertion_statement = assertion_statement
        self.label = label

    def __eq__(self, other):
        return (
            isinstance(other, TheoryAssertionRepresentationWithLabel)
            and self.id == other.id
            and self.theory_statements == other.theory_statements
            and self.assertion_statement == other.assertion_statement
            and self.label == other.label
        )

    def __hash__(self):
        return hash(
            (
                self.id,
                self.theory_statements,
                self.assertion_statement,
                self.label,
            )
        )

    @classmethod
    def from_json(cls, json_dict):
        json_class = json_dict.get("json_class")
        if json_class == "TheoryAssertionRepresentationWithLabel":
            return TheoryAssertionRepresentationWithLabel(
                json_dict["id"],
                json_dict["theory_statements"],
                json_dict["assertion_statement"],
                json_dict.get("label"),
            )
        return None

    def to_json(self):
        return {
            "json_class": "TheoryAssertionRepresentationWithLabel",
            "id": self.id,
            "theory_statements": self.theory_statements,
            "assertion_statement": self.assertion_statement,
            "label": self.label,
        }
