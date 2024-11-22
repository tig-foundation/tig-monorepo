use tig_utils::{
    extend_frontier, pareto_compare, pareto_frontier, pareto_within, scale_frontier, scale_point,
    Frontier, ParetoCompare, PointCompareFrontiers,
};

#[test]
fn test_pareto_compare() {
    assert_eq!(
        pareto_compare(&vec![1, 0], &vec![1, 0]),
        ParetoCompare::Equal
    );
    assert_eq!(
        pareto_compare(&vec![0, 1], &vec![0, 1]),
        ParetoCompare::Equal
    );
    assert_eq!(
        pareto_compare(&vec![1, 1], &vec![0, 1]),
        ParetoCompare::ADominatesB
    );
    assert_eq!(
        pareto_compare(&vec![1, 0], &vec![1, 1]),
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
        pareto_frontier(&points),
        vec![vec![3, 1], vec![2, 2], vec![1, 3]]
            .into_iter()
            .collect::<Frontier>()
    );
}

#[test]
fn test_scale_point() {
    // ceil((x - min + 1) * multiplier)
    assert_eq!(
        scale_point(&vec![3, 1], &vec![0, 0], &vec![10, 10], 1.2),
        vec![4, 2]
    );
    assert_eq!(
        scale_point(&vec![6, 2], &vec![0, 0], &vec![10, 10], 0.7),
        vec![4, 2]
    );
}

#[test]
fn test_scale_frontier() {
    let frontier: Frontier = vec![vec![3, 1], vec![2, 2], vec![0, 4]]
        .into_iter()
        .collect();
    assert_eq!(
        scale_frontier(&frontier, &vec![0, 0], &vec![10, 10], 1.2),
        vec![vec![4, 2], vec![3, 3], vec![1, 5]]
            .into_iter()
            .collect::<Frontier>()
    );
    assert_eq!(
        scale_frontier(&frontier, &vec![0, 0], &vec![10, 10], 0.6),
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
        extend_frontier(&frontier, &vec![0, 0], &vec![10, 10]),
        vec![vec![3, 1], vec![2, 2], vec![0, 4], vec![4, 0]]
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
        pareto_within(&vec![4, 4], &frontier1, &frontier2),
        PointCompareFrontiers::Within
    );
    assert_eq!(
        pareto_within(&vec![4, 0], &frontier1, &frontier2),
        PointCompareFrontiers::Within
    );
    assert_eq!(
        pareto_within(&vec![5, 4], &frontier1, &frontier2),
        PointCompareFrontiers::Above
    );
    assert_eq!(
        pareto_within(&vec![1, 2], &frontier1, &frontier2),
        PointCompareFrontiers::Below
    );
}
