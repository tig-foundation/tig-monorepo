use rand::Rng;
use std::cmp::min;
use std::collections::HashSet;

pub type Point = Vec<i32>;
pub type Frontier<P = Point> = HashSet<P>;

#[derive(Debug, Clone, PartialEq)]
pub enum PointCompareFrontiers<P> {
    Below(P),
    Within,
    Above(P),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParetoCompare {
    ADominatesB,
    Equal,
    BDominatesA,
}

pub trait PointOps {
    type Point;

    fn pareto_compare(&self, other: &Self) -> ParetoCompare;
    fn scale(&self, min_point: &Self, max_point: &Self, multiplier: f64) -> Self::Point;
    fn within(
        &self,
        lower_frontier: &Frontier<Self::Point>,
        upper_frontier: &Frontier<Self::Point>,
    ) -> PointCompareFrontiers<Self::Point>;
}
pub trait FrontierOps {
    type Point;

    fn pareto_frontier(&self) -> Frontier<Self::Point>;
    fn extend(&self, min_point: &Self::Point, max_point: &Self::Point) -> Frontier<Self::Point>;
    fn scale(
        &self,
        min_point: &Self::Point,
        max_point: &Self::Point,
        multiplier: f64,
    ) -> Frontier<Self::Point>;
    fn sample<T: Rng>(&self, rng: &mut T) -> Self::Point;
}

impl PointOps for Point {
    type Point = Point;

    fn pareto_compare(&self, other: &Self) -> ParetoCompare {
        let mut a_dominate_b = false;
        let mut b_dominate_a = false;
        for (a_val, b_val) in self.iter().zip(other) {
            if a_val < b_val {
                b_dominate_a = true;
            } else if a_val > b_val {
                a_dominate_b = true;
            }
        }
        if a_dominate_b == b_dominate_a {
            ParetoCompare::Equal
        } else if a_dominate_b {
            ParetoCompare::ADominatesB
        } else {
            ParetoCompare::BDominatesA
        }
    }
    fn scale(
        &self,
        min_point: &Self::Point,
        max_point: &Self::Point,
        multiplier: f64,
    ) -> Self::Point {
        self.iter()
            .enumerate()
            .map(|(i, value)| {
                // Calculate the offset for the current dimension
                let offset = ((value - min_point[i] + 1) as f64) * multiplier;
                // Scale the point and clamp it between min_point and max_point
                (min_point[i] + offset.ceil() as i32 - 1).clamp(min_point[i], max_point[i])
            })
            .collect()
    }
    fn within(
        &self,
        lower_frontier: &Frontier<Self::Point>,
        upper_frontier: &Frontier<Self::Point>,
    ) -> PointCompareFrontiers<Self::Point> {
        // Check if the point is not dominated by any point in the lower frontier
        if let Some(point) = lower_frontier
            .iter()
            .find(|lower_point| self.pareto_compare(lower_point) == ParetoCompare::BDominatesA)
        {
            return PointCompareFrontiers::Below(point.clone());
        }

        // Check if the point does not dominate any point in the upper frontier
        if let Some(point) = upper_frontier
            .iter()
            .find(|upper_point| self.pareto_compare(upper_point) == ParetoCompare::ADominatesB)
        {
            return PointCompareFrontiers::Above(point.clone());
        }

        PointCompareFrontiers::Within
    }
}

impl FrontierOps for Frontier {
    type Point = Point;

    fn pareto_frontier(&self) -> Frontier<Self::Point> {
        let mut frontier = self.clone();

        for point in self.iter() {
            if !frontier.contains(point) {
                continue;
            }

            let mut dominated_points = HashSet::new();
            for other_point in frontier.iter() {
                match point.pareto_compare(other_point) {
                    ParetoCompare::ADominatesB => {
                        dominated_points.insert(other_point.clone());
                    }
                    ParetoCompare::BDominatesA => {
                        dominated_points.insert(point.clone());
                        break;
                    }
                    ParetoCompare::Equal => {}
                }
            }
            frontier = frontier.difference(&dominated_points).cloned().collect();
        }

        frontier
    }
    fn extend(&self, min_point: &Self::Point, max_point: &Self::Point) -> Frontier<Self::Point> {
        let mut frontier = self.clone();
        (0..min_point.len()).into_iter().for_each(|i| {
            let mut d = min_point.clone();
            if let Some(v) = frontier.iter().map(|d| d[i]).max() {
                d[i] = v;
            }
            if !frontier.contains(&d) {
                d[i] = min(d[i] + 1, max_point[i]);
                frontier.insert(d);
            }
        });
        frontier
    }
    fn scale(
        &self,
        min_point: &Self::Point,
        max_point: &Self::Point,
        multiplier: f64,
    ) -> Frontier<Self::Point> {
        self.iter()
            .map(|point| point.scale(min_point, max_point, multiplier))
            .collect()
    }
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Point {
        // FIXME only works for 2 dimensional points
        // Potential strategy for >2d: triangulate -> sample triangle -> sample point in triangle
        match self.iter().next() {
            None => panic!("Frontier is empty"),
            Some(point) => {
                if point.len() != 2 {
                    panic!("Only 2 dimensional points are supported");
                }
            }
        };
        // randomly pick a dimension
        let dim = (rng.next_u32() % 2) as usize;
        let dim2 = (dim + 1) % 2;

        // sort points by that dimension
        let mut sorted_points: Vec<&Point> = self.iter().collect();
        sorted_points.sort_by(|a, b| a[dim].cmp(&b[dim]));

        // sample value in that dimension
        let min_v = sorted_points.first().unwrap()[dim];
        let max_v = sorted_points.last().unwrap()[dim];
        let rand_v = rng.gen_range(min_v..=max_v);

        // interpolate value in the other dimension
        match sorted_points.binary_search_by(|point| point[dim].cmp(&rand_v)) {
            Ok(idx) => sorted_points[idx].clone(),
            Err(idx) => {
                let a = sorted_points[idx - 1];
                let b = sorted_points[idx];
                let ratio = (rand_v - a[dim]) as f64 / (b[dim] - a[dim]) as f64;
                let rand_v2 = (a[dim2] as f64 + ratio * (b[dim2] - a[dim2]) as f64).ceil() as i32;
                // a is smaller than b in dim, but larger in dim2
                if rand_v2 == a[dim2] {
                    a.clone()
                } else {
                    (0..2)
                        .into_iter()
                        .map(|i| if i == dim { rand_v } else { rand_v2 })
                        .collect()
                }
            }
        }
    }
}
