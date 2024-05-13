use tig_utils::{Frontier, FrontierOps, ParetoCompare, PointCompareFrontiers, PointOps};

#[test]
fn test_pareto_compare() {
    assert_eq!(vec![1, 0].pareto_compare(&vec![1, 0]), ParetoCompare::Equal);
    assert_eq!(vec![1, 0].pareto_compare(&vec![0, 1]), ParetoCompare::Equal);
    assert_eq!(
        vec![1, 1].pareto_compare(&vec![0, 1]),
        ParetoCompare::ADominatesB
    );
    assert_eq!(
        vec![1, 0].pareto_compare(&vec![1, 1]),
        ParetoCompare::BDominatesA
    );
}

#[test]
fn test_pareto_frontier() {
    let points: Frontier = vec![
        vec![3, 1],
        vec![1, 0],
        vec![0, 1],
        vec![1, 1],
        vec![0, 0],
        vec![2, 2],
        vec![2, 1],
        vec![1, 3],
    ]
    .into_iter()
    .collect();
    assert_eq!(
        points.pareto_frontier(),
        vec![vec![2, 2], vec![3, 1], vec![1, 3]]
            .into_iter()
            .collect::<Frontier>()
    );
}

#[test]
fn test_scale_point() {
    // ceil((x - min + 1) * multiplier)
    assert_eq!(
        vec![3, 1].scale(&vec![0, 0], &vec![10, 10], 1.2),
        vec![4, 2]
    );
    assert_eq!(
        vec![6, 2].scale(&vec![0, 0], &vec![10, 10], 0.7),
        vec![4, 2]
    );
}

#[test]
fn test_scale_frontier() {
    let frontier: Frontier = vec![vec![3, 1], vec![2, 2], vec![0, 4]]
        .into_iter()
        .collect();
    assert_eq!(
        frontier.scale(&vec![0, 0], &vec![10, 10], 1.2),
        vec![vec![4, 2], vec![3, 3], vec![1, 5]]
            .into_iter()
            .collect::<Frontier>()
    );
    assert_eq!(
        frontier.scale(&vec![0, 0], &vec![10, 10], 0.6),
        vec![vec![1, 1], vec![0, 2]]
            .into_iter()
            .collect::<Frontier>()
    );
}

#[test]
fn test_extend() {
    let frontier: Frontier = vec![vec![3, 1], vec![2, 2], vec![0, 4]]
        .into_iter()
        .collect();
    assert_eq!(
        frontier.extend(&vec![0, 0], &vec![10, 10]),
        vec![vec![4, 0], vec![3, 1], vec![2, 2], vec![0, 4]]
            .into_iter()
            .collect::<Frontier>()
    );
}

#[test]
fn test_within() {
    let frontier1: Frontier = vec![vec![3, 1], vec![2, 2], vec![0, 4]]
        .into_iter()
        .collect();
    let frontier2: Frontier = vec![vec![6, 0], vec![5, 3], vec![0, 7]]
        .into_iter()
        .collect();
    assert_eq!(
        vec![4, 4].within(&frontier1, &frontier2),
        PointCompareFrontiers::Within
    );
    assert_eq!(
        vec![4, 0].within(&frontier1, &frontier2),
        PointCompareFrontiers::Within
    );
    assert_eq!(
        vec![5, 4].within(&frontier1, &frontier2),
        PointCompareFrontiers::Above
    );
    assert_eq!(
        vec![1, 2].within(&frontier1, &frontier2),
        PointCompareFrontiers::Below
    );
}
