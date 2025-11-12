# Vehicle Routing with Time Windows
The Vehicle Routing Problem with Time Windows (VRPTW) problem is an established extension of [The classic Vehicle Routing Problem (VRP)](https://en.wikipedia.org/wiki/Vehicle_routing_problem), distinguished by the introduction of time window constraints for each customer, adding a temporal dimension to the already intricate tasks of fleet sizing, route planning, and capacity management. These additional constraints make the VRPTW a better reflection of real-world logistical challenges and opens up a broader landscape for algorithmic innovation. The presence of time windows makes the problem computationally more challenging and encourages the exploration of novel algorithmic frameworks.

## Challenge Formulation
VRPTW involves determining a set of cost-effective routes for a fleet of identical vehicles operating from a single depot to serve a geographically dispersed set of customers. Each vehicle has a fixed capacity and each customer has a known demand for goods and a defined time window during which service must begin. If a vehicle arrives before this time window, it must wait; if it arrives after, service is considered infeasible. The primary objective is to minimise the total distance the fleet must travel to deliver goods to all customers and return to the depot, such that:

1. Each customer is visited by exactly one vehicle,
2. The total demand serviced by each vehicle does not exceed its capacity,
3. Each vehicle starts and ends its route at the depot,
4. Vehicles must arrive at each customer at latest by their `due_time`,
5. Vehicles wait until a customer's `ready_time` if they arrive early,
6. After arriving at a customer, the vehicle only leaves after a `service_time`,
7. Each vehicle must return to the depot at latest by the depot's `due_time`, and
8. The number of vehicles utilised is less than a set fleet size

**Notes:** 
* Each unit of distance takes the equivalent amount of time to travel. i.e. a vehicle will take 100 time units to reach a customer that is 100 distance units away.

## Example

The following is an example of the Vehicle Routing Problem with Time Windows with configurable difficulty. Two parameters can be adjusted in order to vary the difficulty of the challenge instance:

- Parameter 1: $num\textunderscore{ }nodes$ is the number of customers (plus 1 depot) which are distributed across a grid of 1000x1000 with the depot at the centre (500, 500).  
- Parameter 2: $better\textunderscore{ }than\textunderscore{ }baseline$ (see Our Challenge)

Demand of each customer is selected independently and uniformly at random from the range [1, 35]. Each customer is assigned a time window between which they must be serviced. Service duration is set to a fixed value of 10 time units per customer. The maximum capacity of each vehicle is set to 200.


Consider an example instance with `num_nodes=8`:

```
# A sample generated example
CUSTOMER
CUST NO.  XCOORD.   YCOORD.   DEMAND    READY TIME  DUE TIME   SERVICE TIME
       0       500       500         0            0      2318             0
       1        75       250        10            0       868            10
       2       940       582        11          825       884            10
       3       398       419        22            0       682            10
       4       424       690         6          256       273            10
       5       143       482        19          674       717            10
       6       187       292        27            0      1785            10
       7       382       204         3            0       832            10
       8       465       274        25         1386      1437            10
       
max_capacity = 200 # total demand for each route must not exceed this number
fleet_size = 4 # the total number of routes must not exceed this number
baseline_total_distance = 3875
```

The depot is the first node (node 0) with demand 0. The vehicle capacity is set to 200 and the fleet capacity to 4.

Now consider the following routes:

```
Route 1: [0, 6, 1, 7, 8, 0]
Route 2: [0, 4, 2, 0]
Route 3: [0, 3, 5, 0]
```

When evaluating these routes, each route has demand less than 200, the number of vehicles used, 3, is less than the fleet capacity, the time windows are not violated, and the total distance is shorter than 3100, thereby these routes are a solution:

* Route 1: 
    * Depot -> 6 -> 1 -> 7 -> 8 -> Depot
    * Demand = 27 + 10 + 3 + 25 = 65
    * Distance = 376 + 120 + 310 + 109 + 229 = 1144
* Route 2: 
    * Depot -> 4 -> 2 -> Depot
    * Demand = 6 + 11 = 17
    * Distance = 205 + 527 + 448 = 1180
* Route 3: 
    * Depot -> 3 -> 5 -> Depot
    * Demand = 22 + 19 = 41
    * Distance = 130 + 263 + 357 = 750
* Total Distance = 1144 + 1180 + 750 = 3074

These routes are 20.6% better than the baseline: 
```
better_than_baseline = 1 - total_distance / baseline_total_distance 
                     = 1 - 3074 / 3875 
                     = 0.206
```

## Our Challenge
In TIG, the baseline route is determined by using Solomon's I1 insertion heuristic that iteratively inserts customers into routes based on a cost function that balances distance and time constraints. The routes are built one by one until all customers are served. 

Each instance of TIG's vehicle routing problem contains 16 random sub-instances, each with its own baseline routes and baseline distance. For each sub-instance, we calculate how much your routes' total distance is shorter than the baseline distance, expressed as a percentage improvement. This improvement percentage is called `better_than_baseline`. Your overall performance is measured by taking the root mean square of these 16 `better_than_baseline` percentages. To pass a difficulty level, this overall score must meet or exceed the specified difficulty target.

For precision, `better_than_baseline` is stored as an integer where each unit represents 0.1%. For example, a `better_than_baseline` value of 22 corresponds to 22/1000 = 2.2%.

## Applications
* **Logistics & Delivery Services:** Optimizes parcel and ship routing by ensuring vehicles meet customer and operational time constraints, reducing operational costs and environmental impact [^1].
* **E-Commerce & Last-Mile Delivery:** Enables precise scheduling for tight delivery windows in online retail, boosting customer satisfaction and operational efficiency [^2][^3].
* **Healthcare & Home Services:** Schedules mobile healthcare visits within set time slots, enhancing care quality and resource utilization [^4].
* **Waste Collection & Disposal:** Streamlines municipal waste collection by routing vehicles to service areas within prescribed time windows, minimizing fuel use and costs. [^5].
* **Public Transportation & Paratransit:** Coordinates demand-responsive transit and school bus routes, ensuring timely pickups and drop-offs for users [^6]. 
* **Emergency Response & Disaster Relief:** Routes emergency vehicles and supplies to critical sites within urgent time frames, optimizing response times and resource allocation [^1].

[^1]: Toth, P., & Vigo, D. (Eds.). (2014). Vehicle routing: problems, methods, and applications. *Society for industrial and applied mathematics*.
[^2]: Giuffrida, N., Fajardo-Calderin, J., Masegosa, A. D., Werner, F., Steudter, M., & Pilla, F. (2022). Optimization and machine learning applied to last-mile logistics: A review. *Sustainability, 14*(9), 5329.
[^3]: Zennaro, I., Finco, S., Calzavara, M., & Persona, A. (2022). Implementing E-commerce from logistic perspective: Literature review and methodological framework. *Sustainability, 14*(2), 911.
[^4]: Du, G., Liang, X., & Sun, C. (2017). Scheduling optimization of home health care service considering patients’ priorities and time windows. *Sustainability, 9*(2), 253.
[^5]: Babaee Tirkolaee, E., Abbasian, P., Soltani, M., & Ghaffarian, S. A. (2019). Developing an applied algorithm for multi-trip vehicle routing problem with time windows in urban waste collection: A case study. *Waste Management & Research, 37*(1_suppl), 4-13.
[^6]: Huang, A., Dou, Z., Qi, L., & Wang, L. (2020). Flexible route optimization for demand-responsive public transit service. *Journal of Transportation Engineering, Part A: Systems, 146*(12), 04020132.