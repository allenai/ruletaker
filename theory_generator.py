import argparse
import common
from common import Example, Fact, Rule, Theory, TheoryAssertionInstance
import json

import nltk
from nltk import Nonterminal, PCFG
from numpy.random import choice
import random

import problog
from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
from problog.engine import NonGroundProbabilisticClause, UnknownClause
from problog.engine_stack import NegativeCycle
from problog.formula import LogicFormula, LogicDAG
from problog.sdd_formula import SDD

from tqdm.auto import tqdm

import utils
from utils import parse_fact, parse_rule


class TheoremProverConfig:
    """Config for the theory generation, read from a json input config file.
    Sample theory config:
    {
        "theory": {
            "num_examples": 200,
            "statement_types_per_example": [ {
                    "start_symbol": "Fact",
                    "num_statements_range": [1, 16]
                },
                {
                    "start_symbol": "Rule",
                    "num_statements_range": [1, 8]
                }
            ],
            "theorem_prover": {
                "fact_nonterminals": ["Fact"],
                "rule_nonterminals": ["Rule"],
                "predicate_nonterminals": ["Attribute", "Relation"],
                "variable_nonterminals": ["Variable"],
                "constant_nonterminals": ["Entity"]
            }
        }
    }
    """

    def __init__(self, grammar, **config_args):
        def expand_nonterminal(nonterminal, grammar):
            """Generate sentences for a given grammar production, identified by the LHS Nonterminal.
            Return a collection of strings."""
            productions = [
                item
                for item in grammar.productions()
                if item.lhs().symbol() == nonterminal
            ]
            generated_sentences_for_noterminal = []
            for production in productions:
                rhs = production.rhs()
                sentences = []
                for item in rhs:
                    if isinstance(item, Nonterminal):
                        curr_sentences = expand_nonterminal(item.symbol(), grammar)
                    else:
                        curr_sentences = [item]
                    if len(sentences) == 0:
                        sentences += curr_sentences
                    else:
                        new_sentences = []
                        for sentence in sentences:
                            for curr_sentence in curr_sentences:
                                new_sentences.append(f"{sentence} {curr_sentence}")
                        sentences = new_sentences
                    generated_sentences_for_noterminal.extend(sentences)
            return generated_sentences_for_noterminal

        def initialize_terms(grammar):
            """Enumerate all the terms- predicates, variables and constants in the given grammar."""
            for nt in self.predicate_nonterminals:
                predicate_terms = expand_nonterminal(nt, grammar)
                predicate_terms = [utils.predicatize(term) for term in predicate_terms]
                self.predicates = predicate_terms
            for nt in self.variable_nonterminals:
                variable_terms = expand_nonterminal(nt, grammar)
                variable_terms = [utils.variablize(term) for term in variable_terms]
                self.variables = variable_terms
            for nt in self.constant_nonterminals:
                constant_terms = expand_nonterminal(nt, grammar)
                constant_terms = [utils.constantize(term) for term in constant_terms]
                self.constants = constant_terms

        self.grammar = grammar
        self.predicates = []
        self.variables = []
        self.constants = []
        self.fact_nonterminals = []
        self.rule_nonterminals = []
        self.predicate_nonterminals = []
        self.variable_nonterminals = []
        self.constant_nonterminals = []
        for key, value in config_args.items():
            setattr(self, key, value)
        initialize_terms(grammar)


def choose_production(grammar, nonterminal):
    """Choose a production with specified nonterminal as LHS based on the probability distibution
    of the grammar."""
    productions = [
        item for item in grammar.productions() if item.lhs().symbol() == nonterminal
    ]
    if len(productions) == 0:
        raise ValueError(f"Nonterminal {nonterminal} not found in the grammar!")
    probabilities = [production.prob() for production in productions]
    chosen_production = choice(productions, p=probabilities)
    return chosen_production


def generate_random_statement(grammar, nonterminal, theorem_prover_config):
    """Generate a random statement from the given nonterminal LHS in the grammar."""
    chosen_production = choose_production(grammar, nonterminal)
    rhs = chosen_production.rhs()
    sentence = ""
    for item in rhs:
        if isinstance(item, Nonterminal):
            item_generated_statement = generate_random_statement(
                grammar, item.symbol(), theorem_prover_config
            )
        else:
            if nonterminal in theorem_prover_config.predicate_nonterminals:
                item = utils.predicatize(item)
            elif nonterminal in theorem_prover_config.variable_nonterminals:
                item = utils.variablize(item)
            elif nonterminal in theorem_prover_config.constant_nonterminals:
                item = utils.constantize(item)
            item_generated_statement = item
        if len(sentence) > 0:
            sentence += " "
        sentence += item_generated_statement
    return sentence


def get_min_proof_depth(source_id, node_id_map, curr_depth):
    """Get the depth of the minimum-depth proof."""
    # Does a depth-first traversal as children are found, incrementing the depth if a conjunctive
    # node is found, i.e., two nodes are being combined.
    # If the node with children is a conjunction (indicates chaining), returns the maximum depth
    # of the paths from each of the children.
    # It it is a disjunction (indicates alternate proof paths), returns the minimum depth of the
    # paths from each of the children.
    if source_id not in node_id_map:
        return curr_depth
    node = node_id_map.get(source_id, None)
    if isinstance(node, problog.formula.atom):
        return curr_depth
    if isinstance(node, problog.formula.conj):
        curr_depth += 1
    depths_at_children = []
    for child_id in node.children:
        child_node = node_id_map.get(child_id, None)
        depths_at_children.append(
            get_min_proof_depth(child_id, node_id_map, curr_depth)
        )
    if isinstance(node, problog.formula.conj):
        return max(depths_at_children)
    return min(depths_at_children)


def get_proof_depth(sdd):
    """Walk the returned SDD structure from ProbLog to get an integer representing the depth
    of the proof."""

    # Extract the queried assertion in the example.
    # We assume a single query per sdd as that is how we are generating examples.
    # Example of what a query looks like: (is_young(erin), 7).
    # This is a tuple containing the name of the node in the proof tree representing
    # the assertion, and its id.
    queries = list(sdd.queries())
    query = queries[0]
    query_node_id = query[1]

    # Construct a map of node ids to node objects for lookup during traversal.
    # Example of a map that can be built:
    # {  1: atom(identifier=0, probability=1.0, group=None, name=is_white(erin), source=None),
    #    2: atom(identifier=(6, (erin,) {{}}, 0), probability=1.0, group=(6, (erin,) {{}}),
    #       name=choice(6,0,is_young(erin),erin), source=None),
    #    3: conj(children=(1, 2), name=None),
    #    4: atom(identifier=2, probability=1.0, group=None, name=is_small(erin), source=None),
    #    5: atom(identifier=(14, (erin,) {{}}, 0), probability=1.0, group=(14, (erin,) {{}}),
    #       name=choice(14,0,is_young(erin),erin), source=None),
    #    6: conj(children=(4, 5), name=None),
    #    7: disj(children=(3, 6), name=is_young(erin))
    # }
    node_ids_and_nodes = [(node_id, node) for node_id, node, _ in iter(sdd)]
    node_id_map = dict(node_ids_and_nodes)

    min_proof_depth = get_min_proof_depth(query_node_id, node_id_map, 0)
    return min_proof_depth


def run_theory_in_problog(theory, assertion):
    """Run the given theory and assertion through ProbLog engine to obtain a True/False label.
    If an exception is encountered, return None so that this example will not be part of output."""
    theorem_prover = "problog"
    sdd = None
    try:
        program = theory.program(theorem_prover, assertion)
        lf = LogicFormula.create_from(program)  # ground the program
        dag = LogicDAG.create_from(lf)  # break cycles in the ground program
        sdd = SDD.create_from(dag)
        result = sdd.evaluate()
        result_tuples = [(k, v) for k, v in result.items()]
        if len(result_tuples) == 0:
            label = False
        else:
            label = result_tuples[0][1] != 0.0
        return (label, sdd)
    except (NegativeCycle, NonGroundProbabilisticClause, UnknownClause) as e:
        return (None, sdd)


def get_truth_label_proof_and_proof_depth(
    theory, assertion, theorem_prover_config, theorem_prover
):
    """Get a truth label for a given theory and assertion by running it through
    specified theorem prover."""
    label, proof_depth, proof = None, None, None
    if theorem_prover.lower() == "problog":
        label, sdd = run_theory_in_problog(theory, assertion)
        if label is not None and sdd is not None:
            proof_depth = get_proof_depth(sdd)
            # The SDD object's string representation looks something like this:
            # 1: atom(identifier=14, probability=1.0, group=None, name=big('Erin'), source=None)
            # 2: atom(identifier=(34, ('Erin',) {{}}, 0), probability=1.0, group=(34, ('Erin',) {{}}), name=choice(34,0,cold('Erin'),'Erin'), source=None)
            # 3: conj(children=(1, 2), name=None)
            # 4: atom(identifier=(42, ('Erin',) {{}}, 0), probability=1.0, group=(42, ('Erin',) {{}}), name=choice(42,0,round('Erin'),'Erin'), source=None)
            # 5: conj(children=(3, 4), name=round('Erin'))
            # Queries :
            # * round('Erin') : 5 [query]
            # When written as a string field (`proof` field of a TheoryAssertionInstance
            # into the output json with the below formatting, it will look like this:
            # "proof": "1: atom(identifier=15, probability=1.0, group=None, name=green('Harry'), source=None) |
            #  2: atom(identifier=(24, ('Harry',) {{}}, 0), probability=1.0, group=(24, ('Harry',) {{}}),
            #     name=choice(24,0,young('Harry'),'Harry'), source=None) |
            #  3: conj(children=(1, 2), name=young('Harry')) |
            #  Queries : * young('Harry') : 3 [query]""
            proof = (
                str(sdd)
                .replace("\n*", "")
                .replace("\n", " | ")
                .rstrip(" | ")
                .lstrip(" | ")
            )
    return label, proof_depth, proof


def generate_random_example(
    example_id,
    example_id_prefix,
    grammar,
    theorem_prover_config,
    statement_types,
    assertion_start_symbol,
    theorem_prover,
):
    example = None

    generated_facts = set()
    generated_rules = set()
    generated_statements = set()

    predicates_in_rule_consequents = set()
    arguments_in_generated_statements = set()

    # Generate examples for every required type of statement (Start Symbol type)
    for statement_type in statement_types:
        start_symbol = statement_type["start_symbol"]
        num_statements_range = statement_type["num_statements_range"]
        req_num_statements = random.randint(
            num_statements_range[0], num_statements_range[1]
        )
        num_generated_statements = 0
        max_generation_attempts = 20
        num_generation_attempts = 0
        while num_generated_statements < req_num_statements:
            generated_statement = generate_random_statement(
                grammar, start_symbol, theorem_prover_config
            )
            if generated_statement in generated_statements:
                if num_generation_attempts == max_generation_attempts:
                    break
                num_generation_attempts += 1
            else:
                if start_symbol in theorem_prover_config.rule_nonterminals:
                    # If the current start symbol for generation is supposed to be a rule,
                    # parse the generated statement as a rule.
                    generated_rule = parse_rule(generated_statement)

                    # Constrain rule such that:
                    #   All non-first entities appear earlier in the rule.
                    # This means that if the randomly generated rule does NOT conform to
                    # this requirement, then retry. If not, it is a valid rule, so update
                    # the set of generated statements by adding the rule.
                    rule_constraint_statisfied = True
                    first_entity = generated_rule.lhs[0].arguments[0]
                    first_fact_remaining_arguments = generated_rule.lhs[0].arguments[1:]
                    remaining_arguments = []
                    for fact in generated_rule.lhs[1:]:
                        remaining_arguments.extend(fact.arguments)
                    remaining_arguments.extend(generated_rule.rhs.arguments)
                    used_entities = set()
                    used_entities.add(first_entity)
                    for entity in remaining_arguments:
                        if entity not in used_entities:
                            rule_constraint_statisfied = False
                            break
                        else:
                            used_entities.add(entity)

                    if rule_constraint_statisfied:
                        generated_rules.add(generated_rule)
                        for f in generated_rule.lhs:
                            arguments_in_generated_statements.update(f.arguments)
                        predicates_in_rule_consequents.add(generated_rule.rhs.predicate)
                        arguments_in_generated_statements.update(
                            generated_rule.rhs.arguments
                        )
                        generated_statements.add(generated_statement)
                        num_generated_statements += 1
                        num_generation_attempts = 0

                elif start_symbol in theorem_prover_config.fact_nonterminals:
                    # If the current start symbol for generation is supposed to be a fact,
                    # parse the generated statement as a fact.
                    generated_fact = parse_fact(generated_statement)
                    generated_facts.add(generated_fact)
                    arguments_in_generated_statements.update(generated_fact.arguments)
                    generated_statements.add(generated_statement)
                    num_generated_statements += 1
                    num_generation_attempts = 0

    theory = Theory(
        list(generated_facts), list(generated_rules), list(generated_statements)
    )

    # Constrain the generation of the assertion so that:
    # 1. A statement in the theory is not just repeated as an assertion as is.
    # 2. The assertion is valid only if it contains arguments that appear somewhere in the theory and
    #    a predicate that appears on the RHS of some rule.
    assertion = None
    max_valid_assertion_generation_attempts = 20
    num_attempts = 0
    generated_valid_assertion = False
    while not generated_valid_assertion and (
        num_attempts < max_valid_assertion_generation_attempts
    ):
        assertion_statement = generate_random_statement(
            grammar, assertion_start_symbol, theorem_prover_config
        )
        assertion = parse_fact(assertion_statement)
        if (
            assertion_statement not in generated_statements
            and assertion.predicate in predicates_in_rule_consequents
            and len(
                set(assertion.arguments).intersection(arguments_in_generated_statements)
            )
            > 0
        ):
            generated_valid_assertion = True
        num_attempts += 1

    if assertion is not None:
        # Plug the generated statements into theorem prover
        label, proof_depth, proof = get_truth_label_proof_and_proof_depth(
            theory, assertion, theorem_prover_config, theorem_prover
        )

        if label is not None:
            # Construct example with label
            example_id = str(example_id)
            if len(example_id_prefix) > 0:
                example_id = f"{example_id_prefix}-{example_id}"
            example = Example(
                example_id,
                TheoryAssertionInstance(
                    theory=theory,
                    assertion=assertion,
                    label=label,
                    min_proof_depth=proof_depth,
                    proof=proof,
                ),
            )

    return example


def generate_theory(
    grammar,
    config,
    theory_op_file,
    theorem_prover,
):
    """Generate a theory with specified properties per config file specifications, using the
    specified grammar.
    Arguments:
    theory_op_file: Output jsonl file containing the generated examples.
    """
    # Get Theorem Prover Config and initialize Theorem Prover
    theorem_prover_config = TheoremProverConfig(
        grammar, **config["theory"]["theorem_prover"]
    )

    num_examples = config["theory"]["num_examples"]
    min_num_positive_examples = config["theory"]["min_num_positive_examples"]
    max_num_negative_examples = num_examples - min_num_positive_examples
    statement_types = config["theory"]["statement_types_per_example"]
    assertion_start_symbol = config["assertion"]["start_symbol"]
    example_id_prefix = config.get("example_id_prefix", "")

    # Generate examples for every required type of statement (Start Symbol type)
    num_true_labels = 0
    num_false_labels = 0
    curr_num_examples = 0
    progress_tracker = tqdm(total=num_examples)
    progress_tracker.set_description(desc="Generating Examples...")
    while (
        curr_num_examples < num_examples and num_true_labels < min_num_positive_examples
    ):
        example = generate_random_example(
            curr_num_examples + 1,
            example_id_prefix,
            grammar,
            theorem_prover_config,
            statement_types,
            assertion_start_symbol,
            theorem_prover,
        )
        if example is not None:
            if example.theory_assertion_instance.label:
                num_true_labels += 1
            else:
                if num_false_labels == max_num_negative_examples:
                    continue
                else:
                    num_false_labels += 1
            json.dump(example.to_json(), theory_op_file)
            theory_op_file.write("\n")
            curr_num_examples += 1
            progress_tracker.update()
    progress_tracker.close()
    print(f"Generated {curr_num_examples} examples.")
    print(f"  No. with True label: {num_true_labels}")
    print(f"  No. with False label: {num_false_labels}")


def preprocess_pcfg(grammar_file):
    """Preprocesses given PCFG grammar file to return a collection of strings representing
    all the productions in the grammar. Expected grammar file format: NLTK PCFG format,
    for e.g.:
        Statement -> Fact
        Fact -> Polarity '(' Attribute Entity ')'
        Entity -> 'cat' | 'dog' | 'bald eagle' | 'rabbit' | 'mouse'
        Attribute -> 'red' | 'blue' | 'green' | 'kind' | 'nice' | 'big'
        Polarity -> '+' [0.8] | '-' [0.2]
    """
    # Iterate through the lines and collect productions in a dictionary, keyed by
    # the nonterminals. So if there are two lines, one with S -> NP VP | VP and another
    # with S -> NP VP PP on two different lines, the dictionary will contain a key 'S'
    # with value 'NP VP | VP | NP VP PP'.
    productions = []
    nonterminal_dict = {}
    for line in grammar_file.readlines():
        production_parts = line.strip().split("->", 1)
        if len(production_parts) == 2:
            lhs = production_parts[0].strip()
            rhs = production_parts[1]
            if lhs not in nonterminal_dict:
                nonterminal_dict[lhs] = rhs
            else:
                nonterminal_dict[lhs] += " | " + rhs

    # Iterate through the productions and check if each possible RHS has a probability
    # associated with it, expected to be specified like [0.5].
    productions = []
    for nonterminal in nonterminal_dict:
        rhs = nonterminal_dict[nonterminal]
        rhs_parts = [rhs_part.strip() for rhs_part in rhs.split("|")]
        num_parts = len(rhs_parts)
        found_probs = True
        for rhs_part in rhs_parts:
            rhs_part_items = rhs_part.split(" ")
            rhs_part_last_item = rhs_part_items[-1]
            if not (
                rhs_part_last_item.startswith("[") and rhs_part_last_item.endswith("]")
            ):
                found_probs = False
                break
        # If any of the RHS part items did not have an associated probability, assign all of them equal
        # probability.
        if not found_probs:
            prob = 1.0 / num_parts
            rhs_parts_with_probs = []
            for rhs_part in rhs_parts:
                rhs_part_mod = rhs_part + " " + "[" + str(prob) + "]"
                rhs_parts_with_probs.append(rhs_part_mod)
            rhs_parts = rhs_parts_with_probs
        final_rhs = " | ".join(rhs_parts)
        production = f"{nonterminal} -> {final_rhs}"
        productions.append(production)
    return productions


def main():
    parser = argparse.ArgumentParser(description="Theory Generator.")
    parser.add_argument("--grammar", required=True, help="Grammar (CFG) for theory")
    parser.add_argument(
        "--config-json",
        required=True,
        help="Json format config file with parameters to generate theory",
    )
    parser.add_argument(
        "--op-theory-jsonl",
        help="Output Jsonl file containing an example json object per line. Json object has the format of the TheoryAssertionInstance class",
    )
    parser.add_argument(
        "--theorem-prover",
        choices=common.supported_theorem_provers,
        default=common.default_theorem_prover,
        help="Thorem proving engine to use. Only supported one right now is problog.",
    )
    args = parser.parse_args()

    with open(args.grammar, "r") as grammar_file, open(
        args.config_json, "r"
    ) as config_json_file:
        theory_op_file = open(args.op_theory_jsonl, "w")
        config = json.load(config_json_file)
        production_strs = preprocess_pcfg(grammar_file)
        grammar_str = "\n".join(production_strs)
        grammar = PCFG.fromstring(grammar_str)
        generate_theory(
            grammar,
            config,
            theory_op_file,
            args.theorem_prover,
        )


if __name__ == "__main__":
    main()
