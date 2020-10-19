#!/usr/bin/env python
import copy
import random

variables = ['X']

class Fact:
    """Class to represent a simple fact in a theory. It basically consists of a predicate
    with its arguments, a polarity (positive/negaitve) for it, along with an associated
    probability.""" 
    def __init__(self, polarity, predicate, arguments, prob=1.0):
        self.polarity = polarity
        self.predicate = predicate
        self.arguments = arguments
        self.probability = prob

    def constants(self):
        return set([argument for argument in self.arguments if argument.islower()])

    def __repr__(self):
        return f'({self.polarity} {self.predicate}({", ".join(self.arguments)}))'
 
    #def formatted_arguments(self):
    #    """Format a predicate argument. This essentially means, if an argument is a constant
    #    (terminal) keep it as is. If it represents a variable, e.g. "something", replace it
    #    with the right representation for a variable in logical form. Here we use a variable from
    #    the collection 'variables' defined at the top."""
    #    arguments_mod = []
    #    for arg in self.arguments:
    #        arg_mod = arg
    #        if arg not in variables:
    #            arg_mod = variables[0]
    #        else:
    #            arg_mod = arg.lower()
    #        arguments_mod.append(arg_mod)
    #    return arguments_mod    

    def logical_form(self, theorem_prover, standalone=True):
        """Produce a logical form representation of the fact in specified theorem prover format."""
        lf = ''
        arguments = self.arguments
        if theorem_prover.lower() == 'problog':
            prob = f'{self.probability}::'   
            if self.polarity != '+':
                lf += '\+' 
            lf += f'{self.predicate}({", ".join(arguments)})'
            if standalone:
                lf = f'{prob}{lf}.'
        elif theorem_prover.lower() == 'pydatalog':
            args_txt = ', '.join(arguments)
            lf = None
            if self.polarity == '+':
                lf = f'{self.predicate}({args_txt})'
            else:
                lf = f'~{self.predicate}({args_txt})'
            if standalone:
                lf = f'+{lf}' 
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
            return arg.lstrip("'").rstrip("'")

        fact_nl = ""
        if len(self.arguments) == 1:
            arg = format_argument_as_nl(self.arguments[0])
            fact_nl = f"{arg} is {self.predicate}"
        elif len(self.arguments) == 2:
            arg1 = format_argument_as_nl(self.arguments[0])
            arg2 = format_argument_as_nl(self.arguments[1])
            fact_nl = f"{arg1} {self.predicate} {arg2}"
        if len(fact_nl) > 0:
            if self.polarity != '+':
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

    def constants(self):
        facts = self.lhs + [self.rhs]
        constants_in_rule = set()
        for fact in facts:
            constants_in_rule = constants_in_rule.union(fact.constants())
        return constants_in_rule
    
    def __repr__(self):
        lhs_repr = f'({" ".join(str(lhs_part) for lhs_part in self.lhs)})'
        return f'{lhs_repr} -> {self.rhs}'

    def logical_form(self, theorem_prover):
        """Produce a logical form representation of the rule in specified theorem prover format."""
        lf = ''
        if theorem_prover.lower() == 'problog':
            prob = f'{self.probability}::'
            antecedant_lf = ', '.join([lhs_fact.logical_form(theorem_prover, False) for lhs_fact in self.lhs])
            consequent_lf = self.rhs.logical_form(theorem_prover, False)
            lf = f'{prob}{consequent_lf} :- {antecedant_lf}.'
        elif theorem_prover.lower() == 'pydatalog':
            fact_txts = []
            for fact in self.lhs:
                fact_txts.append(fact.logical_form(theorem_prover, False))
            antecedant = ' & '.join(fact_txts)
            consequent = self.rhs.logical_form(theorem_prover, False) 
            lf = f'{consequent} <= {antecedant}'
        return lf

    def nl(self):
        """Produce a simple English representation of the rule.
        The LHS Facts are each converted to NL and joined together with 'and's in the middle.
        NL is generated for the RHS Fact. Then the two are joined together with the template
        If <LHS> then <RHS>.
        """
        lhs_nl_statements = [f.nl(standalone=False) for f in self.lhs]
        lhs_nl = ' and '.join(lhs_nl_statements)
        rhs_nl = self.rhs.nl(standalone=False)
        nl = f'If {lhs_nl} then {rhs_nl}.'
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
        prog = '\n'.join(fact_lfs + rule_lfs)    
        if theorem_prover == 'problog' and \
            assertion is not None:
                assertion_lf = assertion.logical_form(theorem_prover, standalone=False)
                prog += f'\nquery({assertion_lf}).'
        return prog

    def program_with_assertion(self, theorem_prover, assertion):
        """The program along with assertion in theorem proving engine's expected format."""
        if theorem_prover == 'problog':
            return self.program(theorem_prover, assertion)
        elif theorem_prover == 'pydatalog':
            program = self.program(theorem_prover)
            assertion_lf = assertion.logical_form(theorem_prover)
            program += f'\nask({assertion_lf})'
            return program 
        else:
            return self.program(theorem_prover, assertion)

    def nl(self):
        fact_nls = [f.nl() for f in self.facts]
        rule_nls = [r.nl() for r in self.rules]
        nl = ' '.join(fact_nls + rule_nls)
        return nl        


class TheoryAssertionInstance:
    """Class representing a single example for a model. Consists of a theory with a corresponding
    assertion and a gold truth label for the assertion's truthiness wrt the theory."""
    def __init__(self, theory, assertion, label=None):
        self.theory = theory
        self.assertion = assertion
        self.label = label