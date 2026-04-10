## overview
part 1: refactoring of tests
part 2: cleaning code
part 3: fixing issues found during refactoring and cleaning


### part 1: refactoring of tests

step 1: create new directory for tests

step 2: determine conceptual groupings of tests:
- tokenization modes: bpe vs concept, discrete vs continuous, factored vs fused
- distribution selection and overrides
- overrides and autos: i.e. auto bpe/concept, auto level, fused requiring concept
- milestones: inpatient, outpatient, the specific trigger time etc.
- age: make sure age token works well
- milestone emission/age emession with time gaps: make sure corret milestones included
- bpe behaviro with different vocab size requests and special character counts (what if vocab size is less than number of special tokens? etc.)

** if any issues are found in the tests, fix them


### part 2: cleaning code
Now that the tests are in place, can do some cleaning and refactoring.

- evaluate code for inefficiecies and remove any redundant code that was produced during development
- consider removal of minimal features that add bloat (like subsampling the df to do bpe training)
- determine if we should break this larget file into 2-3 smaller files. check with me before doing this with a specific plan

