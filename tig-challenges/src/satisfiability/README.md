# Boolean Satisfiability

[The SAT (or Boolean Satisfiability) problem is a decision problem in computer science](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem). It's the problem of determining if there exists a truth assignment to a given Boolean formula that makes the formula true (satisfies all clauses).

A Boolean formula is built from:

- Boolean variables: $x_1, x_2, x_3, \ldots$
- Logical connectives: AND ($\land$), OR ($\lor$), NOT ($\neg$)
- Parentheses for grouping: ( )

3-SAT is a special case of SAT where each clause is limited to exactly three literals (a literal is a variable or its negation). An example with 4 variables and 3 clauses can be seen below:

$$(x_1 \lor x_2 \lor x_3) \land (\neg x_1 \lor \neg x_3 \lor \neg x_4) \land (\neg x_2 \lor x_3 \lor x_4)$$

For this particular example, one possible truth assignment that satisfies this formula is $x_1 = True$, $x_2 = False$, $x_3 = True$, $x_4 = False$. This can be verified by substituting the variables and evaluating that every clause will result in $True$.

# Example

The following is an example of the 3-SAT problem with configurable difficulty. Two parameters can be adjusted in order to vary the difficulty of the challenge instance:

- Parameter 1: $num\textunderscore{ }variables$ = **The number of variables**.  
- Parameter 2: $clauses\textunderscore{ }to\textunderscore{ }variables\textunderscore{ }percent$ = **The number of variables as a percentage of the number of clauses**. 

The number of clauses is derived from the above parameters.

$$num\textunderscore{ }clauses = floor(num\textunderscore{ }variables \cdot \frac{clauses\textunderscore{ }to\textunderscore{ }variables\textunderscore{ }percent}{100})$$

Where $floor$ is a function that rounds a floating point number down to the closest integer.

Consider an example instance with `num_variables=4` and `clauses_to_variables_percent=75`:

```
clauses = [
    [1, 2, -3],
    [-1, 3, 4],
    [2, -3, 4]
]
```

Each clause is an array of three integers. The absolute value of each integer represents a variable, and the sign represents whether the variable is negated in the clause (negative means it's negated).

The clauses represents the following Boolean formula:

```
(X1 or X2 or not X3) and (not X1 or X3 or X4) and (X2 or not X3 or X4)
```

Now consider the following assignment:

```
assignment = [False, True, True, False]
```

This assignment corresponds to the variable assignment $X1=False, X2=True, X3=True, X4=False$.

When substituted into the Boolean formula, each clause will evaluate to True, thereby this assignment is a solution as it satisfies all clauses.

# Our Challenge
In TIG, the 3-SAT Challenge is based on the example above with configurable difficulty.  Please see the challenge code for a precise specification. 

# Applications

SAT has a vast range of applications in science and industry in fields including computational biology, formal verification, and electronic circuit design. For example:

SAT is used in computational biology to solve the "cell formation problem" of [organising a plant into cells](https://www.sciencedirect.com/science/article/abs/pii/S0957417412006173).
SAT is also heavily utilised in [electronic circuit design](https://dl.acm.org/doi/abs/10.1145/337292.337611).

<img src="../images/circuit.jfif" alt="Application of SAT" width="100%"/>

<figcaption>Figure 1: Chips made possible by electronic circuit design.</figcaption>
<br/>
