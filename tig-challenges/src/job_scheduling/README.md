# Job Scheduling

The Job Scheduling challenge is based on [Flexible Job Shop (FJSP)](https://en.wikipedia.org/wiki/Flexible_job_shop_scheduling): given a set of jobs, each assigned to a product with a fixed sequence of operations, schedule all operations on eligible machines to minimize the **makespan** (the maximum completion time across all jobs). Each operation can run on a subset of machines, and each eligible machine has its own processing time for that operation.

## Challenge Overview

For our challenge, we use flexible job-shop instances with configurable difficulty. Parameters that affect difficulty include:

- **Number of jobs** and **number of machines** – larger instances increase the search space.
- **Machine flexibility** – how many machines are eligible per operation (JSSP-style vs FJSP-style).
- **Product variety** – one shared route vs multiple products with different operation sequences.
- **Routing structure** – strict sequential order vs mixed or random operation order per product.

**Constraints.** A valid solution must satisfy:

- `job_schedule.len()` equals `num_jobs`; each job has one entry.
- Each job’s schedule length matches its product’s number of operations.
- Each `(machine_id, start_time)` uses an eligible machine for that operation.
- Operations for a job are sequential: each start time ≥ previous operation’s finish time.
- No machine processes overlapping operations.

**Objective.** The goal is to minimize makespan (last finish time of all jobs)

## Example

Consider an instance with 4 jobs, 3 machines, 3 operation types, and 2 products. Product 0 uses op0 and op1; product 1 uses op1 and op2. Op0 can run on machines 0 and 1; op1 on 0, 1, 2; op2 only on machine 2.

```
num_jobs = 4
num_machines = 3
num_operations = 3   # three operation types: op0, op1, op2
jobs_per_product = [2, 2]   # jobs 0–1 are product 0, jobs 2–3 are product 1

# product_processing_times[product][op] = map: machine_id -> processing_time
# Product 0: op0, op1.  Product 1: op1, op2.
# op0: machines 0,1.  op1: machines 0,1,2.  op2: machine 2 only.
product_processing_times = [
    [ {0: 3, 1: 4}, {0: 2, 1: 1, 2: 3} ],   # product 0: Op0, Op1
    [ {0: 2, 1: 1, 2: 3}, {2: 4} ],         # product 1: Op1, Op2
]
```

A feasible solution:

```
job_schedule = [
    [(0, 0), (1, 4)],   # Job 0 (product 0): Op0 on machine 0, Op1 on machine 1
    [(1, 0), (0, 4)],   # Job 1 (product 0): Op0 on machine 1, Op1 on machine 0
    [(2, 0), (2, 3)],   # Job 2 (product 1): Op1 on machine 2, Op2 on machine 2
    [(1, 5), (2, 7)],   # Job 3 (product 1): Op1 on machine 1, Op2 on machine 2
]
```

Verification:

- **Job 0:** Op0 on machine 0 from 0→3, Op1 on machine 1 from 4→5 → completion time 5.
- **Job 1:** Op0 on machine 1 from 0→4, Op1 on machine 0 from 4→6 → completion time 6.
- **Job 2:** Op1 on machine 2 from 0→3, Op2 on machine 2 from 3→7 → completion time 7.
- **Job 3:** Op1 on machine 1 from 5→6, Op2 on machine 2 from 7→11 → completion time 11.

No machine has overlapping operations; each job’s operations are sequential. The **makespan** is max(5, 6, 7, 11) = **11**.

## Our Challenge

In TIG, your algorithm does not return a solution; it calls `save_solution` as it runs. The **last** saved solution is evaluated. A valid solution must meet all constraints above; invalid solutions are not scored.

The evaluated metric is **quality** (a fixed-point integer with 6 decimal places), comparing your makespan to a greedy dispatching-rule baseline: `quality = 1.0 - make_span / greedy_makespan`

Higher quality is better. See the challenge code for the precise encoding.

**Problem types your solver should handle:**

- **Strict:** Fixed machine assignments (flexibility = 1.0, JSSP-style), single shared route, strict sequential order (flow_structure = 0.0), 20% reentrance, moderate product variety.
- **Parallel:** Multiple eligible machines per operation (flexibility = 3.0, FJSP-style), single shared route, strict order, 20% reentrance, moderate product variety.
- **Random:** Fixed machine assignments (flexibility = 1.0), multiple routes with mixed routing (flow_structure = 0.4), no reentrance, high product variety.
- **Complex:** Multiple eligible machines (flexibility = 3.0), multiple routes with mixed routing (flow_structure = 0.4), 20% reentrance, high product variety.
- **Chaotic:** Very high machine flexibility (flexibility = 10.0), many routes with full randomization (flow_structure = 1.0), no reentrance, high product variety.
