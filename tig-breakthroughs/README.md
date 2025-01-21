# tig-breakthroughs

A folder that hosts submissions of algorithmic methods made by Innovators in TIG.

Each submissions is committed to their own branch with the naming pattern:

`<challenge_name>\breakthrough\<method_name>` 

## Making a Submission

1. Read important docs for what is a Breakthrough and how it is rewarded:
    * [Implementations vs Breakthroughs](../docs/guides/breakthroughs.md)

    * [Voting Guidelines for Token Holders](../docs/guides/voting.md)

2. Email the following to `breakthroughs@tig.foundation` with subject "Breakthrough Submission (`<breakthrough_name>`)":
    * **Evidence form**: copy & fill in [`evidence.md`](./evidence.md). Of particular importance is Section 1 which describes your breakthrough

    * **Invention assignment**: copy & replace [invention_assignment.doc](../docs/agreements/invention_assignment.doc) the highlighted parts. Inventor and witness must sign.

    * **Address Signature**: use [etherscan](https://etherscan.io/verifiedSignatures#) to sign a message `I am signing this message to confirm my submission of breakthrough <breakthrough_name>`. Use your player_id that is making the submission. Send the verified etherscan link with message and signature.

    * (Optional) **Code implementation**: attach code implementing your breakthrough. Do not submit this code to TIG separately. This will be done for you

**Notes**:
* The time of submission will be taken as the timestamp of the auto-reply to your email attaching the required documents.

* Iterations are permitted for errors highlighted by the Foundation. This will not change the timestamp of your submission

* 250 TIG will be deducted from your Available Fee Balance to make a breakthrough submission

* An additional 10 TIG will be deducted from your Available Fee Balance to make an algorithm submission (if one is attached)

* You can topup via the [Benchmarker page](https://play.tig.foundation/benchmarker)

## Method Submission Flow

1. New submissions get their branch pushed to a private version of this repository
2. A new submission made during round `X` will have its branch pushed to the public version of this repository at the start of round `X + 2`
3. From the start of round `X + 3` till the start of round `X + 4`, token holders can vote on whether they consider the method to be a breakthrough based off the submitted evidence
4. At the start of round `X + 4`, if the submission has at least 50% yes votes, it becomes active
5. Every block, a method's adoption is the sum of all algorithm adoption, where the algorithm is attributed to that method. Methods with at least 50% adoption earn rewards and a merge point
6. At the end of a round, a method from each challenge with the most merge points, meeting the minimum threshold of 5040, gets merged to the `main` branch
    * Merged methods are considered breakthroughs, and receive rewards every block where their adoption is greater than 0%