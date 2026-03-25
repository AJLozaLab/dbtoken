##Test cases:

we need to test the following edge cases. please design a test_edge.py single file without fancy imports that tests these cases:

- multiple patients in a row: correctly verify eos and sos insertion
- nested inpatient encounters: don't go back to OP time until the last encounter is closed
- numeric (class,text_value) that was not selected as concept when fused numeric is selected
- orderding of simultaneous events. should be maintained as in the source data in the token stream


before programming, come up with other edge cases