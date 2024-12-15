# tig-breakthroughs

A folder that hosts submissions of algorithmic methods made by Innovators in TIG.

Each submissions is committed to their own branch with the naming pattern:

`<challenge_name>\method\<method_name>` 

## Making a Submission

1. Read important docs for what is a Breakthrough and how it is rewarded:
    * [Implementations vs Breakthroughs](../docs/guides/breakthroughs.md)

    * [Voting Guidelines for Token Holders](../docs/guides/voting.md)

2. Copy `template.md` from the relevant challenges folder (e.g. [`knapsack/tempate.md`](./knapsack/template.md)), and fill in the details with evidence of your breakthrough

3. Copy [invention_assignment.doc](../docs/agreements/invention_assignment.doc), fill in the details, and sign

4. Email your invention assignment to contact@tig.foundation with subject "Invention Assignment"

5. Submit your evidence via https://play.tig.foundation/innovator
    * 250 TIG will be deducted from your Available Fee Balance to make a submission
    * You can topup via the [Benchmarker page](https://play.tig.foundation/benchmarker)

## Method Submission Flow

1. New submissions get their branch pushed to a private version of this repository
2. A new submission made during round `X` will have its branch pushed to the public version of this repository at the start of round `X + 2`
3. From the start of round `X + 2` till the start of round `X + 4`, token holders can vote on whether they consider the method to be a breakthrough based off the submitted evidence
4. At the start of round `X + 4`, if the submission has at least 50% yes votes, it becomes active
5. Every block, a method's adoption is the sum of all algorithm adoption, where the algorithm is attributed to that method. Methods with at least 50% adoption earn rewards and a merge point
6. At the end of a round, a method from each challenge with the most merge points, meeting the minimum threshold of 5040, gets merged to the `main` branch
    * Merged methods are considered breakthroughs, and receive rewards every block where their adoption is greater than 0%