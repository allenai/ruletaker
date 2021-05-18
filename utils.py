import random
from common import Fact, Rule


def variablize(term):
    """Formats a term (string) to look like a Variable."""
    # Replace spaces by underscores and uppercase first letter of a Variable term
    term = term.replace(" ", "_")
    return term[0].upper() + term[1:]


def predicatize(term):
    """Formats a term (string) to look like a Predicate."""
    # Replace spaces by underscores and lowercase first letter of a Predicate term
    term = term.replace(" ", "_")
    return term[0].lower() + term[1:]


def constantize(term):
    """Formats a term (string) to look like a Constant."""
    # Replace spaces by underscores and enclose in quotes for a Constant term
    return f"'{term.replace(' ', '_')}'"


def parse_fact(fact_txt):
    """Parses text into Fact.
    E.g.:
    - ( red X )
    - ( eats X Anne )
    """
    invalid_format_msg = (
        f"Invalid statement format: {fact_txt}! "
        + "Expected fact in the format: +/- ( predicate argument1 argument2 etc. )"
    )
    fact_txt = fact_txt.strip()
    polarity = None
    predicate = None
    arguments = []
    polarity = fact_txt[0]
    if polarity != "+" and polarity != "-":
        raise ValueError(invalid_format_msg)
    fact_txt = fact_txt[1:].strip()
    if fact_txt.startswith("(") and fact_txt.endswith(")"):
        fact_predicate_args_txt = fact_txt[1:][:-1].strip()
        fact_tokens = fact_predicate_args_txt.split(" ")
        if len(fact_tokens) < 2:
            raise ValueError(invalid_format_msg)
        else:
            fact_tokens = [token.strip() for token in fact_tokens]
            predicate = fact_tokens[0]
            arguments = fact_tokens[1:]
    else:
        raise ValueError(invalid_format_msg)
    return Fact(polarity, predicate, arguments)


def parse_multiple_facts(multiple_facts_txt):
    """Parses text containing a list of facts.
    E.g.:
    [ - red ( X ) , - eats ( X  Anne ) ]
    """
    facts = []
    multiple_facts_txt = multiple_facts_txt.strip()
    if multiple_facts_txt.startswith("[") and multiple_facts_txt.endswith("]"):
        multiple_facts_txt = multiple_facts_txt[1:][:-1].strip()
        multiple_fact_txts = multiple_facts_txt.split(",")
        for fact_txt in multiple_fact_txts:
            fact = parse_fact(fact_txt)
            facts.append(fact)
    else:
        fact_txt = multiple_facts_txt.strip()
        fact = parse_fact(fact_txt)
        facts.append(fact)
    return facts


def parse_rule(statement_txt):
    potential_rule_parts = statement_txt.split("->", 1)
    lhs_txt = potential_rule_parts[0].strip()
    rhs_txt = potential_rule_parts[1].strip()
    lhs = None
    try:
        lhs = parse_multiple_facts(lhs_txt)
    except ValueError:
        raise ValueError(f"Unable to parse statement {statement_txt} as a rule.")
    rhs = parse_fact(rhs_txt)
    return Rule(lhs, rhs)


def parse_statement(statement_txt):
    # Tries to parse the given text as a rule. Expected format e.g.:
    # [ - red ( X ) , - eats ( X  Anne ) ] -> - eats ( Anne , Charlie )
    # lhs is made up of a list of facts and rhs is made up of a
    # single fact.
    potential_rule_parts = statement_txt.split("->", 1)
    if len(potential_rule_parts) == 2:
        return parse_rule(statement_txt)
    else:
        return parse_fact(statement_txt)
