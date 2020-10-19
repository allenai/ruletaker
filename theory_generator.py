import argparse
import csv
import common
from common import Fact, Rule, Theory, TheoryAssertionInstance
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


import pyDatalog
from pyDatalog import pyDatalog
from pyDatalog.pyDatalog import load

import utils
from utils import parse_fact, parse_statement

class TheoremProverConfig:
    """Config for the theory generation, read from a json input config file.
    Sample config:
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
                "predicate_nonterminals": ["Attribute", "Relation"],
                "variable_nonterminals": ["Variable"],
                "constant_nonterminals": ["Entity"]
            } 
        },
        "assertion": {
            "start_symbol": "Fact"
        }
    }
    """
    def __init__(self, grammar, **config_args):

        def expand_nonterminal(nonterminal, grammar):
            """Generate sentences for a given grammar production, identified by the LHS Nonterminal.
            Return a collection of strings."""
            productions = [item for item in grammar.productions() if item.lhs().symbol() == nonterminal]
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
                                new_sentences.append(f'{sentence} {curr_sentence}')
                        sentences = new_sentences
                    generated_sentences_for_noterminal.extend(sentences)
            return generated_sentences_for_noterminal

        def initialize_terms(grammar):
            """Enumerate all the terms- predicates, variables and constants in the given grammar."""
            for nt in self.predicate_nonterminals:
                predicate_terms = expand_nonterminal(nt, grammar)
                predicate_terms = [ utils.predicatize(term) for term in predicate_terms ]
                self.predicates = predicate_terms
            for nt in self.variable_nonterminals:
                variable_terms = expand_nonterminal(nt, grammar)
                variable_terms = [ utils.variablize(term) for term in variable_terms ]
                self.variables = variable_terms
            for nt in self.constant_nonterminals:
                constant_terms = expand_nonterminal(nt, grammar)
                constant_terms = [ utils.constantize(term) for term in constant_terms ]
                self.constants = constant_terms

        self.grammar = grammar
        self.predicates = []
        self.variables = []
        self.constants = []
        self.predicate_nonterminals = []
        self.variable_nonterminals = []
        self.constant_nonterminals = []
        for key, value in config_args.items():
            setattr(self, key, value)
        initialize_terms(grammar)
 

def choose_production(grammar, nonterminal):
    """Choose a production with specified nonterminal as LHS based on the probability distibution
    of the grammar.""" 
    productions = [item for item in grammar.productions() if item.lhs().symbol() == nonterminal]
    if len(productions) == 0:
        raise ValueError(f'Nonterminal {nonterminal} not found in the grammar!')
    probabilities = [production.prob() for production in productions]
    chosen_production = choice(productions, p=probabilities)
    return chosen_production


def initialize_theorem_prover(theorem_prover_config, theorem_prover):
    """Intialize the theorem proving engine. Depending on the chosen theorem prover, nothing may
    need to be done."""
    # Initializes pyDatalog by creating necessary terms for predicates and variables.
    if theorem_prover == 'pydatalog':
        pyDatalog.clear()
        # Create theory terms in pyDatalog
        terms_arg = f'\'{", ".join(theorem_prover_config.predicates + theorem_prover_config.variables)}\''
        pyDatalog.create_terms(terms_arg)


def generate_random_statement(grammar, nonterminal, theorem_prover_config):
    """Generate a random statement from the given nonterminal LHS in the grammar."""
    chosen_production = choose_production(grammar, nonterminal)
    rhs = chosen_production.rhs()
    sentence = ''
    for item in rhs:
        if isinstance(item, Nonterminal):
            item_generated_statement = generate_random_statement(grammar, item.symbol(), theorem_prover_config)
        else:
            if nonterminal in theorem_prover_config.predicate_nonterminals:
                item = utils.predicatize(item)
            elif nonterminal in theorem_prover_config.variable_nonterminals:
                item = utils.variablize(item)
            elif nonterminal in theorem_prover_config.constant_nonterminals:
                item = utils.constantize(item)
            item_generated_statement = item
        if len(sentence) > 0:
            sentence += ' '
        sentence += item_generated_statement
    return sentence


def run_theory_in_pydatalog(theory, assertion):
    """Run the given theory and assertion through PyDatalog engine to obtain a True/False label.
    If an exception is encountered, return None so that this example will not be part of output."""
    theorem_prover = 'pydatalog'
    logical_forms = []
    for fact_or_rule in theory.facts + theory.rules:
        if not isinstance(fact_or_rule, Fact) and not isinstance(fact_or_rule, Rule):
            raise ValueError(f'Encountered an object that is not a Fact or a Rule!')
        logical_forms.append(fact_or_rule.logical_form(theorem_prover))
    assertion_logical_form = assertion.logical_form(theorem_prover)
    pydatalog_statements_txt = '\n'.join(logical_forms)
    pyDatalog.clear()
    pyDatalog.load(pydatalog_statements_txt)
    try:
        result = pyDatalog.ask(assertion_logical_form)
        label = len(result.answers) > 0
        return label
    except AttributeError:
        return False


def run_theory_in_problog(theory, assertion):
    """Run the given theory and assertion through ProbLog engine to obtain a True/False label.
    If an exception is encountered, return None so that this example will not be part of output."""
    theorem_prover = 'problog'
    try:
        program = theory.program(theorem_prover, assertion)
        lf = LogicFormula.create_from(program)   # ground the program
        dag = LogicDAG.create_from(lf)     # break cycles in the ground program
        sdd = SDD.create_from(dag)
        result = sdd.evaluate()
        result_tuples = [(k, v) for k, v in result.items()]
        if len(result_tuples) == 0:
            return False
        return (result_tuples[0][1] != 0.0)
    except (NegativeCycle, NonGroundProbabilisticClause, UnknownClause) as e:
        return None
    return None


def get_truth_label(theory, assertion, theorem_prover_config, theorem_prover):
    """Get a truth label for a given theory and assertion by running it through
    specified theorem prover."""
    label = None
    if theorem_prover.lower() == 'pydatalog':
        label = run_theory_in_pydatalog(theory, assertion)
    elif theorem_prover.lower() == 'problog':
        label = run_theory_in_problog(theory, assertion)
    return label


def generate_random_example(
    grammar,
    theorem_prover_config,
    statement_types,
    assertion_start_symbol,
    theorem_prover
):
    example = None

    generated_facts = set()
    generated_rules = set()
    generated_statements = set()

    predicates_in_generated_statements = set()
    arguments_in_generated_statements = set()

    # Generate examples for every required type of statement (Start Symbol type)
    for statement_type in statement_types:
        start_symbol = statement_type['start_symbol']
        num_statements_range = statement_type['num_statements_range']
        req_num_statements = random.randint(num_statements_range[0], num_statements_range[1])
        num_generated_statements = 0
        max_unique_generation_attempts = 10
        num_unique_generation_attempts = 0
        while num_generated_statements < req_num_statements:
            generated_statement = generate_random_statement(grammar, start_symbol, theorem_prover_config)   
            if generated_statement in generated_statements:
                if num_unique_generation_attempts == max_unique_generation_attempts:
                    break
                num_unique_generation_attempts += 1
            else:
                num_unique_generation_attempts = 0
                generated_statement_parsed = parse_statement(generated_statement)
                if isinstance(generated_statement_parsed, Fact):
                    generated_facts.add(generated_statement_parsed)
                    predicates_in_generated_statements.add(generated_statement_parsed.predicate)
                    arguments_in_generated_statements.update(generated_statement_parsed.arguments)
                else:
                    generated_rules.add(generated_statement_parsed)
                    for f in generated_statement_parsed.lhs:
                        predicates_in_generated_statements.add(f.predicate)
                        arguments_in_generated_statements.update(f.arguments)
                    predicates_in_generated_statements.add(generated_statement_parsed.rhs.predicate)
                    arguments_in_generated_statements.update(generated_statement_parsed.rhs.arguments)    
                                        

                generated_statements.add(generated_statement)
                num_generated_statements += 1
   
    theory = Theory(list(generated_facts), list(generated_rules), list(generated_statements))

    # Constrain the generation of the assertion so that:
    # 1. A statement in the theory is not just repeated as an assertion as is.
    # 2. The assertion is valid only if it contains a predicate and arguments that appear somewhere in the theory.
    assertion = None
    max_valid_assertion_generation_attempts = 20
    num_attempts = 0
    generated_valid_assertion = False
    while not generated_valid_assertion and (num_attempts < max_valid_assertion_generation_attempts):
        assertion_statement = generate_random_statement(grammar, assertion_start_symbol, theorem_prover_config)
        assertion = parse_fact(assertion_statement)
        if assertion_statement not in generated_statements and \
            assertion.predicate in predicates_in_generated_statements and \
                len(set(assertion.arguments).intersection(arguments_in_generated_statements)) > 0:
                generated_valid_assertion = True
        num_attempts += 1        

    if assertion is not None:
        # Plug the generated statements into theorem prover
        label = get_truth_label(theory, assertion, theorem_prover_config, theorem_prover) 

        if label is not None:
            # Construct example with label       
            example = TheoryAssertionInstance(theory, assertion, label) 
    
    return example


def generate_theory(grammar, config, theory_lf_file, theory_program_file, theory_english_file, theorem_prover):
    """Generate a theory with specified properties per config file specifications, using the
    specified grammar. 
    Arguments:
    theory_lf_file: Optional output file containing generated theories logical forms. Predicates
    are in prefix notation; e.g.: <polarity> (<predicate> <arg1> <arg2>).
    theory_program_file: Optional output file containing programs for the generated theories in
    the format required for the chosen theorem prover.
    theory_english_file: Optional output file containing generated theories in English.
    """

    # Get Theorem Prover Config and initialize Theorem Prover 
    theorem_prover_config = TheoremProverConfig(grammar, **config["theory"]["theorem_prover"])
    initialize_theorem_prover(theorem_prover_config, theorem_prover)

    theory_logical_forms_writer = None
    theory_program_writer = None
    theory_english_writer = None
    if theory_lf_file is not None:
        theory_logical_forms_writer = csv.writer(theory_lf_file, delimiter=',')
    if theory_program_file is not None:
        theory_program_writer = csv.writer(theory_program_file, delimiter=',')
    if theory_english_file is not None:
        theory_english_writer = csv.writer(theory_english_file, delimiter=',')   

    num_examples = config["theory"]["num_examples"]
    statement_types = config["theory"]["statement_types_per_example"] 
    assertion_start_symbol = config["assertion"]["start_symbol"]
    
    # Generate examples for every required type of statement (Start Symbol type)
    num_true_labels = 0
    num_false_labels = 0
    curr_num_examples = 0
    while curr_num_examples < num_examples:
        example = generate_random_example(grammar, \
            theorem_prover_config, \
            statement_types, \
            assertion_start_symbol, \
            theorem_prover)
        if example is not None:
            if example.label:
                num_true_labels += 1
            else:
                num_false_labels += 1
            if theory_logical_forms_writer is not None:
                theory_logical_forms_writer.writerow([' '.join(example.theory.statements_as_texts), str(example.assertion), example.label])
            program = example.theory.program_with_assertion(theorem_prover, example.assertion)
            if theory_program_writer is not None:
                theory_program_writer.writerow([program, example.label])
            theory_in_nl = example.theory.nl()
            assertion_in_nl = example.assertion.nl()    
            if theory_english_writer is not None:
                theory_english_writer.writerow([theory_in_nl, assertion_in_nl, example.label])    
            curr_num_examples += 1
    print(f'Generated {curr_num_examples} examples.')
    print(f'  No. with True label: {num_true_labels}')
    print(f'  No. with False label: {num_false_labels}')      


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
        production_parts = line.strip().split('->', 1)
        if len(production_parts) == 2:
           lhs = production_parts[0].strip()
           rhs = production_parts[1]
           if lhs not in nonterminal_dict:
               nonterminal_dict[lhs] = rhs
           else:
               nonterminal_dict[lhs] += ' | ' + rhs
  
    # Iterate through the productions and check if each possible RHS has a probability
    # associated with it, expected to be specified like [0.5].
    productions = []
    for nonterminal in nonterminal_dict:
       rhs = nonterminal_dict[nonterminal]   
       rhs_parts = [rhs_part.strip() for rhs_part in rhs.split('|')]
       num_parts = len(rhs_parts)
       found_probs = True
       for rhs_part in rhs_parts:
           rhs_part_items = rhs_part.split(' ')
           rhs_part_last_item = rhs_part_items[-1]
           if not (rhs_part_last_item.startswith('[') and rhs_part_last_item.endswith(']')):
               found_probs = False
               break
       # If any of the RHS part items did not have an associated probability, assign all of them equal
       # probability.
       if not found_probs:
           prob = 1.0 / num_parts
           rhs_parts_with_probs = []
           for rhs_part in rhs_parts:
               rhs_part_mod = rhs_part + ' ' + '[' + str(prob) + ']'
               rhs_parts_with_probs.append(rhs_part_mod)
           rhs_parts = rhs_parts_with_probs
       final_rhs = ' | '.join(rhs_parts)
       production = f'{nonterminal} -> {final_rhs}'
       productions.append(production)
    return productions   


def main():
    parser = argparse.ArgumentParser(description='Theory Generator.')
    parser.add_argument('--grammar', required=True, help='Grammar (CFG) for theory')
    parser.add_argument('--config-json', required=True, help='Json format config file with parameters to generate theory')
    parser.add_argument('--op-theory-logical-form', help='Csv file with the predicates generated from the grammar. Format: <theory-predicates>, <predicate-to-prove>, <label>')
    parser.add_argument('--op-theory-program', help='Csv file with the logic program for each generated theory in the format appropriate for the chosen theorem prover. Format: <program>, <label>')
    parser.add_argument('--op-theory-english', help='Csv file contaning theories in English, to use for training. Format: <theory-in-english-sentences>, <assertion>, <label: True/False>')
    parser.add_argument('--theorem-prover', required=True, choices=['problog', 'pydatalog'], help='Json format config file with parameters to generate theory')
    args = parser.parse_args()
 
    with open(args.grammar, 'r') as grammar_file, \
        open(args.config_json,'r') as config_json_file:
        theory_lf_file = None
        theory_program_file = None
        theory_english_file = None
        if args.op_theory_logical_form is not None:
            theory_lf_file = open(args.op_theory_logical_form, 'w')
        if args.op_theory_program is not None:    
            theory_program_file = open(args.op_theory_program, 'w')
        if args.op_theory_english is not None:        
            theory_english_file = open(args.op_theory_english, 'w')
        config = json.load(config_json_file)
        production_strs = preprocess_pcfg(grammar_file)
        grammar_str = '\n'.join(production_strs)
        grammar = PCFG.fromstring(grammar_str)
        generate_theory(grammar, config, theory_lf_file, theory_program_file, theory_english_file, args.theorem_prover)


if __name__ == "__main__":
    main()
