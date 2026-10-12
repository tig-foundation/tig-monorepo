# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** epure_imp_v2
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** Thibaut Vidal
* **Unique Algorithm Identifier (UAI):** c002_a110

## Aditional information

Four sizes find shorter routes than the original method. The 900-customer size is unchanged.

- 600 customers: when splitting the map into regions stops helping, the regions get smaller. When it helps a lot, the regions grow again.
- 700 customers: two of the starting route sets are built differently, one from customers near the depot and one by sweeping around the map, and those sets are kept and mixed with the others.
- 800 customers: exchanges between routes look at nearby customers instead of customers with a similar delivery size.
- 1000 customers: after a good route change is found, other changes that are only about half as good are still considered.

Achieves higher quality for the same or better run time than hgs_epure

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses