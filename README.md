# ruletaker

This repo contains tools and utilities to:
1. Generate datasets of theories and assertions meant to test the logical reasoning capabilities of a model. For details see the paper [Transformers as Soft Reasoners over Language](https://arxiv.org/abs/2002.05867).
2. Run existing theories through a theorem proving engine to obtain labels.


For any features you might want, or issues, we welcome your pull requests! You can also file issues in this GitHub repository.


## Theory Generator

We provide `theory_generator.py` to generate a dataset of theories and assertions. This helps generate a dataset with gold labels derived by running each theory through a theorem proving engine. The only supported engine currently is [Problog](https://problog.readthedocs.io/en/latest/). Anyone who wishes to use a different theorem prover needs to make appropriate code changes and plug the new engine as a choice to the `theorem_prover` command line argument which currently defualt to `problog`.

The inputs and outputs to this tool and a description of how to use it is described in this section.

### Inputs

1. Grammar: A grammar to generate theories, in PCFG format. E.g.:
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
In order for the theory generation to work properly, there are some key assumptions wrt what the grammar generates:

* A top-level statement can be a Fact or a Rule.
* A Fact contains a predicate with one or more arguments in prefix notation, along with an associated polarity: + / - . 
  Sample Facts:
          Cat is blue: `+ ( blue 'cat' )`
          Dog does not chase cat: `- ( chases dog cat)`
* A Rule contains an antecedant and a consequent and takes the form: <antecedant> -> <consequent>. The antecedant is one or more Facts. The consequent is a single fact.       Sample Rule:
          If something is blue then it is sad: `+ (blue X) -> + (sad X)`
* The entities can represent constants or variables in the grammar.
  
The theory generator needs to meaningfully associate the grammar nonterminals with the entities (constants and variables) and predicates. This mapping is specified via a config file which is described below.

2. Config: A config file in JSON format that contains some hyperparameter values for the theory generator. E.g.
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
    # The assertion to prove, given the theory. Each example in the dataset contains: a theory, an assertion,
    # and a label obtained by running the theory through a theorem proving engine.
    "start_symbol": "Fact"
  }
}
```

### Outputs
The tool can provide the output theories in three forms. A user may choose any/all of these.
1. A CSV file containing the generated theories and assertions in English.

File format:
> \<theory English statements\>, \<assertion English statement\>, \<truth label\>

Sample output:
> Cat is blue. If X is blue then X is red.,Cat is blue.,True   

2. A CSV file containing the generated theories and assertions as a logic program in the format expected by the theorem prover used, along with a truth label.

File format:
> \<logic program\>, \<truth label\>

Sample output:
> "1.0::blue('cat').
>  1.0::red(X) :- blue(X).
>  query(blue('cat')).",True

3. A CSV file containing the theories and assertions directly as generated by expanding the grammar nonterminals. Here statements take logical forms in prefix notation, along with a truth label. The statements generated from the grammar are expected to be of the form: <polarity> ( <predicate> <argument1> <argument2> ... )
  
File format:
> \<newline-delimited theory statement logical forms\>, \<assertion logical form\>, \<truth label\>

Sample output:
>"\+ ( blue X ) -> + ( red X )
  + ( blue 'cat' )",+ ( blue 'cat' ),True


## Label Generator for existing theories

We provide `theory_label_generator.py` to run existing theories and assertions through a theorem proving engine to obtain labels. You can use this to evaluate any theory dataset as long as facts and rules in the theory can be provided in the expected format. The tool takes the given theories with Facts and Rules, and assertions and runs the theorem prover to obtain a label for the theory assertion pair. If gold labels have already been provided, it computes accuracy metrics against them.

### Input

Input can be provided in one of two formats.

#### Format 1
CSV file in the same format as the logical forms output file (csv output file 3) from the `theory_generator` tool above, but with the label field being optional:

File format:
> \<newline-delimited theory statement logical forms\>, \<assertion logical form\>, \[\<truth label\>\]

Sample:
>"\+ ( blue X ) -> + ( red X )
  + ( blue 'cat' )",(+ blue('cat'))

#### Format 2
This is primarily for AI2's internal use. This is the RuleTaker legacy jsonl format. Sample jsonl snippet is below (this only includes fields that are relevant to the tool).

```
{ "id": "AttNoneg-D3-319", ...
         "triples":{
            "triple1": {
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
```

### Output
The output format depends on the chosen input format (as described in the Inputs section above).

#### Format 1
CSV file with the same format as input file, with an additional field with the theorem prover's generated label.

Sample:
>"\+ ( blue X ) -> + ( red X )
  + ( blue 'cat' )",+ ( blue 'cat' ), True

#### Format 2
This is a jsonl format that has the same structure as the input above,  with an additional Boolean field called '<theorem_prover>_label', for e.g., `problog_label`.


## Running the tools

### Preparing the Python environment

To run any of the tools in the repo, first create a python environment with the necessary dependencies-

```
pip install -r requirements.txt
```

Then use the following command line to run the theory generator. Currently we are using Problog as the underlying theorem proving engine. That is the recommended engine to use.

NOTE:
If you run into [this](https://github.com/wannesm/PySDD/issues/19) issue where the PySDD install is not recognized and once you try to run the program (below), you get an error like:

```
The SDD library is not available. Please install the PySDD package."
problog.errors.InstallError: The SDD library is not available. Please install the PySDD package..
```

To get around this, run the following to ensure that PySDD is installed properly with dependency on the right version of `wheel` and other dependencies. This should fix the issue:

```
pip install -vvv --upgrade --force-reinstall --no-binary :all: --no-deps pysdd
```

### Running the theory generator

Once you have prepared the Python environment as described, you can run the theory generator as follows-
```
python theory_generator.py \
  --grammar <grammar_cfg_file_path> \
  --config-json <theory_config_json_file_path> \
  --op-theory-english <theory_in_english_op_file_path> \
  --op-theory-program <theory_as_a logic_program_op_file_path> \
  --op-theory-logical-form <theory_in_generic_prefix_notation_op_file_path>
```

E.g.:
```
python theory_generator.py \
  --grammar grammars_and_config/grammars/ruletaker_grammar_theory1.txt \
  --config-json grammars_and_config/config/ruletaker_theory_generator_config_theory1.json  \
  --op-theory-english ruletaker_theory1_english.csv \
  --op-theory-program ruletaker_theory1_problog_program.csv \
  --op-theory-logical-form ruletaker_theory1_lf.csv
```

Once this completes, you should see a message that looks like:
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
  --input-file <theory_assertion_logical_forms.csv> | <ruletaker_theory_assertions.jsonl>
  --output-file <theory_assertion_logical_forms_with_labels.csv> | <ruletaker_theory_assertions_with_labels.jsonl>
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

This code has been tested and runs on Python 3.7.7 on the following platforms-
Mac OS 10.13, 10.14
Ubuntu 16.04.6 LTS


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
