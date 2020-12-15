# ruletaker

This repo contains tools and utilities to:
1. Generate datasets of theories and assertions meant to test the logical reasoning capabilities of a model. For details see the paper [Transformers as Soft Reasoners over Language](https://arxiv.org/abs/2002.05867).
2. Run existing theories through a theorem proving engine to obtain labels.

It uses [Problog](https://problog.readthedocs.io/en/latest/) as the theorem-proving engine to generate labels for theory-assertion examples.

For any features you might want, or issues, we welcome your pull requests! You can also file issues in this GitHub repository.

## About this release

The original RuleTaker theory generator and inference engine were written in Lisp. This repository is a more stable and re-engineered Python version of that original software. The theory generator here uses a declarative grammar that users can modify to generate different theories with different distributions. For the inference engine, we here use ProbLog, with probabilities 1 and 0 to denote T and F. The ProbLog reasoner generates identical results to the original Lisp inference engine, with one exception: The original RuleTaker datasets accidentally included a few non-stratified, inconsistent theories (for theories with negation). For these, the ProbLog reasoner (correctly) throws an error. We will be releasing the Problog-syntax version of the original RuleTaker theories shortly.

In addition, when ProbLog proves a queried fact, it also returns an SLD tree containing the proof. However, this initial release does not use those SLD trees, thus this software does not return a proof nor proof depth for the query assertions that ProbLog is able to prove.


## Theory Generator

We provide `theory_generator.py` to generate a dataset of theories and assertions. This helps generate a dataset with gold labels derived by running each theory through a theorem proving engine. The only supported engine currently is Problog. Anyone who wishes to use a different theorem prover needs to make appropriate code changes and plug the new engine as a choice to the `theorem_prover` command line argument which currently defaults to `problog`.

The inputs and output of this tool and a description of how to use it is described in this section.

### Inputs

1. Grammar: A grammar to generate theories, in [PCFG format](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar). E.g.:
```
Statement -> Fact | Rule
Fact -> Polarity '(' Attribute Entity ')' [0.3] | Polarity '(' Relation Entity  Entity ')' [0.7]
VFact -> Polarity '(' Attribute VEntity ')' [0.3] | Polarity '(' Relation VEntity Entity ')' [0.7]
VFacts -> '[' VFact ',' VFact ']' [0.5] | '[' VFact ']' [0.5]
Rule -> VFacts '->' VFact [1.0]
VEntity -> Entity [0.2] | Variable [0.8]
Entity -> 'cat' | 'dog'
Variable -> 'X'
Attribute -> 'red'
Relation -> 'chases' | 'eats'
Polarity -> '+' [0.5] | '-' [0.5]

```
In order for the theory generation to work properly, there are some key assumptions with respect to what the grammar generates:

* A top-level statement can be a Fact or a Rule.
* A Fact contains a predicate with one or more arguments in prefix notation, along with an associated polarity: + / - . Sample Facts:
  * Cat is blue: `+ ( blue 'cat' )`
  * Dog does not chase cat: `- ( chases dog cat)`
* A Rule contains an antecedant and a consequent and takes the form: <antecedant> -> <consequent>. The antecedant is one or more Facts. The consequent is a single fact. Sample Rule:
  *  If something is blue then it is sad: `+ (blue X) -> + (sad X)`
* The entities can represent constants or variables in the grammar.
  
The theory generator needs to meaningfully associate the grammar nonterminals with the entities (constants and variables) and predicates. This mapping is specified via a config file which is described below.

2. Config: A config file in JSON format that contains some hyperparameter values for the theory generator. Here's an example, decorated with comments not present in the JSON file:
```
{
  "theory": {
    "num_examples": 200, # The total number of examples (theories) that we want in the dataset.
    
    # Config for each type of top-level statement that will be in the theory.
    "statement_types_per_example": [ {
        # This means that in each example (theory) there will be between 1 and 16 statements that are Facts.
        # (A number randomly chosen in the interval [1, 16].
        "start_symbol": "Fact",
        "num_statements_range": [1, 16] 
      },
      {
        # This means that in each example (theory) there will be between 1 and 8 statements that are Rules.
        # (A number randomly chosen in the interval [1, 8].
        "start_symbol": "Rule",
        "num_statements_range": [1, 8]
      }
    ],
    # Specifies the grammar nonterminals that map to predicates, variables and constants, respectively.
    # If any of these is not applicable, it can be left out of the config.
    "theorem_prover": {     
      # Nonterminals that generate the facts in the theory.
      # If the grammar conforms to expected format, these nonterminals should generate statements that have
      # the format: <+|-> ( <predicate> <argument1> [<argument2> <argument3> ...] )
      # where predicate maps to one of the predicate_nonterminals noted below, and
      # arguments are one of the variable_nonterminals or constant_nonterminals noted below.
      "fact_nonterminals": ["Fact"],

      # Nonterminals that generate the rules in the theory.
      # If the grammar conforms to expected format, these nonterminals should generate statements that have
      # the format: [ <fact1> , <fact2>, <fact3>, ... ] -> <consequent_fact>
      # where fact1, fact2, fact3 and consequent_fact have the expected format for fact_nonterminals
      # as noted above.
      "rule_nonterminals": ["Rule"],

      # Nonterminals that map to predicates in the theory's facts.
      "predicate_nonterminals": ["Attribute", "Relation"],

      # Nonterminals that map to arguments to the predicates (above). These are either constants or variables.
      "variable_nonterminals": ["Variable"],
      "constant_nonterminals": ["Entity"]
    } 
  },
  "assertion": {
    # Each example in the dataset contains: a theory, an assertion to prove given the theory,
    # and a label obtained by running the theory through a theorem proving engine. This specifies
    # the grammar nonterminal that maps to an assertion statement.
    "start_symbol": "Fact"
  },
  # Prefix to use while generating ids for examples in the dataset.
  "example_id_prefix": "ruletaker-problog"
}
```

### Output

The tool provides an output JSONL file with one labeled theory-assertion example as JSON object per line. The object includes representations in three forms: the prefix notation logical forms generated by the grammar, the theory and assertion in natural languages (english sentences), and the logic program that was input to the theorem prover to generate gold label. The field names corresponding to these representations are `logical_forms`,`english` and `logic_program` respectively.

Sample output:
```
{  
  "json_class": "Example",
  "theory_assertion_instance": {
    "json_class": "TheoryAssertionInstance",
    "theory": {
      "json_class": "Theory",
      "facts": [
        {
          "json_class": "Fact",
          "polarity": "+",
          "predicate": "smart",
          "arguments": ["'Anne'"],
          "probability": 1.0
        },
        {
          "json_class": "Fact",
          "polarity": "+",
          "predicate": "nice",
          "arguments": ["'Erin'"],
          "probability": 1.0
        },
        ...
      ],
      "rules": [
        {
          "json_class": "Rule", 
          "lhs": [
            {"json_class": "Fact", "polarity": "+", "predicate": "cold", "arguments": ["X"], "probability": 1.0}
          ],
          "rhs": {"json_class": "Fact", "polarity": "+", "predicate": "round", "arguments": ["X"],
          "probability": 1.0}
        },
        {
          "json_class": "Rule",
          "lhs": [
            {"json_class": "Fact", "polarity": "+", "predicate": "round", "arguments": ["X"], "probability": 1.0}
          ],
          "rhs": {"json_class": "Fact", "polarity": "+", "predicate": "kind", "arguments": ["X"], "probability": 1.0}
        },
        ...
      ]
    },
    "assertion":{
      "json_class": "Fact",
      "polarity": "+",
      "predicate": "kind",
      "arguments": ["'Harry'"],
      "probability": 1.0
    },
    "label": false
  },
  "logical_forms": {
    "json_class": "TheoryAssertionRepresentation", 
    "theory_statements": [
      "[ + ( white X ) ] -> + ( young X )",
      "[ + ( big X ) , + ( green X ) ] -> + ( smart X )",
      "+ ( kind 'Fiona' )", "+ ( smart 'Dave' )",
      "+ ( white 'Erin' )",
      "[ + ( quiet X ) ] -> + ( kind X )",
      "+ ( rough 'Dave' )",
      "[ + ( quiet X ) , + ( big X ) ] -> + ( furry X )",
      "[ + ( round X ) , + ( smart X ) ] -> + ( kind X )",
      "+ ( kind 'Dave' )", "+ ( quiet 'Charlie' )",
      "[ + ( round X ) , + ( red X ) ] -> + ( rough X )",
      "[ + ( quiet X ) , + ( round X ) ] -> + ( kind X )"
    ],
    "assertion_statement": "+ ( young 'Dave')"
  },
  "english": {
    "json_class": "TheoryAssertionRepresentation",
    "theory_statements": [
      "Fiona is kind.",
      "Dave is rough.",
      "Dave is smart.",
      "Charlie is quiet.",
      "Dave is kind.",
      "Erin is white.",
      "If X is white then X is young.",
      "If X is big and X is green then X is smart.",
      "If X is round and X is smart then X is kind.",
      "If X is quiet and X is round then X is kind.",
      "If X is round and X is red then X is rough.",
      "If X is quiet then X is kind.",
      "If X is quiet and X is big then X is furry."
    ],
    "assertion_statement": "Dave is young."
  },
  "logic_program": {
    "problog": {
      "json_class": "TheoryAssertionRepresentation",
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
        "1.0::rough(X) :- round(X), red(X).",
         "1.0::kind(X) :- quiet(X).", "1.0::furry(X) :- quiet(X), big(X)."
      ],
      "assertion_statement": "query(young('Dave'))."
    }
  }
}
```

## Label Generator for existing theories

We provide `theory_label_generator.py` to run existing theories and assertions through a theorem proving engine to obtain labels. You can use this to evaluate any theory dataset as long as facts and rules in the theory can be provided in the expected format. The tool takes the given theories with Facts and Rules, and assertions and runs the theorem prover to obtain a label for the theory assertion pair. If gold labels have already been provided, it computes accuracy metrics against them.

### Input

Input can be provided in one of two formats.

#### Format 1
File in JSONL format, i.e., a JSON object per line with structure as defined by the `TheoryAssertionRepresentationWithLabel` class:
```
{
  "theory_statements": List of theory statement logical forms in prefix notation.
  "assertion_statement": String containing the assertion in prefix notation.
  "label": Optional Boolean label field. If set with value true, then the input object has a gold label specified, to compare accuracy against. Output object will contain a boolean value as returned from the theorem prover or None if the theorem prover threw an exception.
  "exception": Optional String field that is a placeholder for the output object use to report exceptions if any, from the theorem prover.
}
```

Sample input:
```
{
  "json_class": "TheoryAssertionRepresentationWithLabel",
  "theory_statements": [
    "+ ( nice 'Harry' )",
    "[ + ( red X ) ] -> + ( red X )",
    "+ ( big 'Charlie' )",
    "+ ( white 'Erin' )",
    "+ ( green 'Bob' ), [ + ( white X ) , + ( red X ) ] -> + ( nice X )",
    "[ + ( red X ) ] -> + ( cold X )",
    "[ + ( rough X ) ] -> + ( nice X )",
    "[ + ( young X ) ] -> + ( kind X )",
    "[ + ( nice X ) ] -> + ( quiet X )",
    "+ ( blue 'Fiona' )",
    "[ + ( round X ) ] -> + ( young X )",
    "+ ( green 'Erin' )", "+ ( round 'Harry' )",
    "[ + ( nice X ) ] -> + ( green X )",
    "+ ( green 'Gary' )",
  "assertion_statement": "+ ( kind 'Fiona' )",
  "label": false,
  "exception": None
}
```

#### Format 2
This is primarily for AI2's internal use. This is the RuleTaker legacy JSONL format. Sample JSONL snippet is below (this only includes fields that are relevant to the tool).

```
{
    "id":"AttNoneg-D3-319",
    ...
    "triples":{
      "triple1":{
        "text":"Bob is cold.",
        "representation":"(\"Bob\" \"is\" \"cold\" \"+\")"
      },
      "triple2":{
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
                                        -> (\"someone\" \"is\" \"quiet\" \"+\"))"
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
        ...
        "representation":"(\"Gary\" \"is\" \"furry\" \"+\")"
      }
   }
}
```

### Output
The output object will take the structure of the `Example` class, which is the same as the output structure of examples generated by `theory_generator.py`. This structure is documented in detailed along with an output snippet Theory Generator section under Output description.

## Running the tools

### Preparing the Python environment

To run any of the tools in the repo, first create a python environment with the necessary dependencies:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --upgrade --force-reinstall --no-binary :all: --no-deps pysdd
```

Then use the following command line to run the theory generator.

Currently we are using Problog as the underlying theorem proving engine. That is the recommended engine to use.

### Running the theory generator

Once you have prepared the Python environment as described, you can run the theory generator as follows-
```
python theory_generator.py \
  --grammar <grammar_cfg_file_path> \
  --config-json <theory_config_json_file_path> \
  --op-theory-jsonl <theory_op_jsonl_file_path>
```

E.g.:
```
python theory_generator.py \
  --grammar grammars_and_config/grammars/ruletaker_grammar_theory1.txt \
  --config-json grammars_and_config/config/ruletaker_theory_generator_config_theory1.json  \
  --op-theory-jsonl ruletaker_theory1_examples.jsonl
```

This will take a minute to run. Once it completes, you should see a message that looks like:

```
Generated 1000 examples.
  No. with True label: 432
  No. with False label: 568
```


### Running the label generator on existing theories

After preparing the Python environment as described earlier, use the following command line to run `theory_label_generator.py`.

Currently we are using Problog as the underlying theorem proving engine and that is the only supported engine.

```
python theory_label_generator.py \
  --input-file <theory_assertion_representations.jsonl> | <ruletaker_theory_assertions.jsonl>
  --output-file <labeled_output.jsonl>
  [--input-format current|legacy]
  [--report-metrics]
```

Note that `--report-metrics` is an optional flag to get accuracy and timing related metrics at the end of the run. If the flag is specified, and your input dataset includes gold labels, you will see metrics reported as follows:

```
Total no. of examples: 150
  No. true: 75
    No. correct: 73
    No. of exceptions: 6
        No. correct with exceptions: 4
    No. incorrect without exception: 0
  No. false: 75
    No. correct: 72
    No. of exceptions: 7
        No. correct with exceptions: 4
    No. incorrect without exception: 0
Total no. correct: 145
Total no. with exceptions: 13
Accuracy: 96.66666666666667

Failure Breakdown by Exception:
    problog.engine_stack.NegativeCycle: 5

Average theorem proving time per example: 0.025506666666666667 secs
```


# Platform

This code has been tested and runs on Python 3.7.7 on the following platforms:

* Mac OS 10.13, 10.14
* Ubuntu 16.04.6 LTS

# Contact

If you need to directly contact someone for any reason where a GitHub pull request or an issue is not appropriate, you can email sumithrab@allenai.org.


# License

Copyright 2020 Allen Institute for AI

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
