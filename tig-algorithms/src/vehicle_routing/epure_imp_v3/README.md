# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** epure_imp_v3
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** Thibaut Vidal
* **Unique Algorithm Identifier (UAI):** c002_a110

## Additional information

All five sizes find shorter routes than the original method.

- 600 nodes: when splitting the map into regions stops helping, the regions get smaller. When it helps a lot, the regions grow again. Three-node swaps stay on, and a later region result is dropped when it matches one already kept on almost every node.
- 700 nodes: two of the starting route sets are built differently, one from nodes near the depot and one by sweeping around the map, and those sets are kept and mixed with the others. Regions start larger and get smaller as the search goes on.
- 800 nodes: exchanges between routes look at nearby nodes. The search keeps a more varied set of region results, rebuilds each region from its best result, and adds a small extra cost for using more vehicles. Regions shrink when progress stalls, the deepest phases run more often, and capacity and time penalties are lighter inside the regions.
- 900 nodes: similar route sets are blended by walking from one toward the other, and some parents are taken from different groups. The walk is longer, and the far end is kept when it is close to the best stop along the way. Splitting a tour adds a small cost for extra vehicles. Local search looks at the expensive routes first, and at the end the shortest route is emptied when that makes the whole solution cheaper.
- 1000 nodes: after a good route change is found, other changes that are only about half as good are still considered. Parent selection looks at three candidates, and the variety ranking looks at a wider ring of neighbours. If a region search stalls, the starting routes are put back in sooner. If the remaining route sets have collapsed onto each other, that region search stops and keeps the best it already has. When two ways of cutting a tour into routes cost almost the same, the cut closer to the fewest vehicles needed is kept.

Achieves higher quality for the same or better run time than hgs_epure and epure_imp_v2

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