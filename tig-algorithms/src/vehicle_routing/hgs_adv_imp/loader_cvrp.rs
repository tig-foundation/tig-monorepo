use super::problem::{NodeData, Problem};
use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct CVRPLoader;

impl CVRPLoader {
    pub fn load(path: &str) -> Result<Problem> {
        let file = File::open(path).map_err(|e| anyhow!("Cannot open CVRP file '{}': {}", path, e))?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader
            .lines()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("Cannot read CVRP file '{}': {}", path, e))?;

        let mut nb_nodes: Option<usize> = None;
        let mut max_capacity: Option<i32> = None;
        let mut node_coord_pos: Option<usize> = None;
        let mut demand_pos: Option<usize> = None;
        let mut depot_pos: Option<usize> = None;

        for (idx, line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if t.starts_with("DIMENSION") {
                if let Some(v) = parse_last_usize(t) {
                    nb_nodes = Some(v);
                }
            } else if t.starts_with("CAPACITY") {
                if let Some(v) = parse_last_i32(t) {
                    max_capacity = Some(v);
                }
            } else if t.starts_with("NODE_COORD_SECTION") {
                node_coord_pos = Some(idx + 1);
            } else if t.starts_with("DEMAND_SECTION") {
                demand_pos = Some(idx + 1);
            } else if t.starts_with("DEPOT_SECTION") {
                depot_pos = Some(idx + 1);
            }
        }

        let nb_nodes = nb_nodes.ok_or_else(|| anyhow!("CVRP DIMENSION not found"))?;
        let max_capacity = max_capacity.ok_or_else(|| anyhow!("CVRP CAPACITY not found"))?;
        let node_coord_pos = node_coord_pos.ok_or_else(|| anyhow!("CVRP NODE_COORD_SECTION not found"))?;
        let demand_pos = demand_pos.ok_or_else(|| anyhow!("CVRP DEMAND_SECTION not found"))?;
        let depot_pos = depot_pos.ok_or_else(|| anyhow!("CVRP DEPOT_SECTION not found"))?;

        let mut node_positions = vec![(0, 0); nb_nodes];
        for i in 0..nb_nodes {
            let raw = lines
                .get(node_coord_pos + i)
                .ok_or_else(|| anyhow!("Missing CVRP coordinate line {}", i + 1))?;
            let p: Vec<&str> = raw.split_whitespace().collect();
            if p.len() < 3 {
                return Err(anyhow!("Invalid CVRP coordinate line: '{}'", raw));
            }
            let id = p[0].parse::<usize>().map_err(|_| anyhow!("Invalid CVRP node id '{}'", p[0]))?;
            if id != i + 1 {
                return Err(anyhow!("CVRP node numbering should be contiguous from 1"));
            }
            let x = p[1].parse::<i32>().map_err(|_| anyhow!("Invalid CVRP x coordinate '{}'", p[1]))?;
            let y = p[2].parse::<i32>().map_err(|_| anyhow!("Invalid CVRP y coordinate '{}'", p[2]))?;
            node_positions[i] = (x, y);
        }

        let mut demands = vec![0i32; nb_nodes];
        let mut total_demand: i64 = 0;
        for i in 0..nb_nodes {
            let raw = lines
                .get(demand_pos + i)
                .ok_or_else(|| anyhow!("Missing CVRP demand line {}", i + 1))?;
            let p: Vec<&str> = raw.split_whitespace().collect();
            if p.len() < 2 {
                return Err(anyhow!("Invalid CVRP demand line: '{}'", raw));
            }
            let id = p[0].parse::<usize>().map_err(|_| anyhow!("Invalid CVRP demand node id '{}'", p[0]))?;
            if id != i + 1 {
                return Err(anyhow!("CVRP demand numbering should be contiguous from 1"));
            }
            let d = p[1].parse::<i32>().map_err(|_| anyhow!("Invalid CVRP demand '{}'", p[1]))?;
            demands[i] = d;
            total_demand += d as i64;
        }

        let depot_line = lines
            .get(depot_pos)
            .ok_or_else(|| anyhow!("Missing CVRP depot line"))?;
        let depot = depot_line
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("Invalid CVRP depot line"))?;
        if depot != "1" {
            return Err(anyhow!("CVRP depot should be node 1"));
        }

        let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
        for i in 0..nb_nodes {
            for j in 0..nb_nodes {
                let dx = (node_positions[i].0 - node_positions[j].0) as f64;
                let dy = (node_positions[i].1 - node_positions[j].1) as f64;
                distance_matrix[i * nb_nodes + j] = (dx.hypot(dy) + 0.5) as i32;
            }
        }

        let lb_vehicles =
            ((total_demand + max_capacity as i64 - 1) / max_capacity as i64) as usize;
        let nb_vehicles = ((1.3 * (total_demand as f64) / max_capacity as f64).ceil() as usize + 3)
            .max(lb_vehicles);

        let node_data: Vec<NodeData> = (0..nb_nodes)
            .map(|i| NodeData {
                start_tw: 0,
                end_tw: 1_000_000,
                service_time: 0,
                demand: demands[i],
            })
            .collect();
        Ok(Problem {
            seed: [0u8; 32],
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            is_vrptw: false,
            fixed_distance_offset: 0,
            max_capacity,
            distance_matrix,
            node_positions,
            node_data,
        })
    }
}

fn parse_last_usize(s: &str) -> Option<usize> {
    s.split_whitespace().last()?.parse::<usize>().ok()
}

fn parse_last_i32(s: &str) -> Option<i32> {
    s.split_whitespace().last()?.parse::<i32>().ok()
}
