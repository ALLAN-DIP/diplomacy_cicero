# A modified version of DAIDE++
when using `daide_visitor.visit(parse_tree)`, instead of returning a daide string, it returns the English translation of that string.

## Usage

```python3
from daidepp import create_daide_grammar, daide_visitor
grammar = create_daide_grammar(level=130, allow_just_arrangement=True, string_type='all')
message = 'PRP (AND (SLO (ENG)) (SLO (GER)) (SLO (RUS)) (SLO (ENG)) (SLO (GER)) (SLO (RUS)))'
parse_tree = grammar.parse(message)
output = daide_visitor.visit(parse_tree) # object composed of dataclass objects in keywords.py
print(output)
# original output: 
PRP ( AND ( SLO ( ENG ) ) ( SLO ( GER ) ) ( SLO ( RUS ) ) ( SLO ( ENG ) ) ( SLO ( GER ) ) ( SLO ( RUS ) ) )
# modified output:
I propose ENG solo, GER solo, RUS solo, ENG solo, GER solo, and RUS solo  
```

## Optional TODOS
- [x] Suffix periods to all messages (unless they (or questionmarks) are already there)
- [ ] Non-rule-based translations/paraphrases
- [ ] Complete translations for simple tokens (e.g. `ENG` -> `England`)
- [ ] Use first and second person pronouns