use super::genetic::Genetic;
use super::loader_tig::TigLoader;
#[cfg(feature = "benchmark_io")]
use super::loader_cvrp::CVRPLoader;
#[cfg(feature = "benchmark_io")]
use super::loader_vrptw::VRPTWLoader;
use super::params::Params;
use super::problem::Problem;
use super::reverse_mode;
#[cfg(feature = "benchmark_io")]
use anyhow::anyhow;
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{Map, Value};
#[cfg(feature = "benchmark_io")]
use std::fs::{create_dir_all, File};
#[cfg(feature = "benchmark_io")]
use std::io::{BufWriter, Write};
#[cfg(feature = "benchmark_io")]
use std::path::Path;
use std::time::Instant;
use tig_challenges::vehicle_routing::*;

pub struct Solver;

impl Solver {
    #[cfg(feature = "benchmark_io")]
    fn write_solution_to_file(
        path: &str,
        solution: &Solution,
        cost: i32,
        nb_routes: usize,
        cpu_time_secs: f64,
    ) -> Result<()> {
        let out_path = Path::new(path);
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent)?;
            }
        }

        let file = File::create(out_path)?;
        let mut writer = BufWriter::new(file);

        let mut route_id = 1usize;
        for route in &solution.routes {
            let clients: Vec<usize> = route.iter().copied().filter(|&c| c != 0).collect();
            if clients.is_empty() {
                continue;
            }
            write!(writer, "Route #{}:", route_id)?;
            for client in clients {
                write!(writer, " {}", client)?;
            }
            writeln!(writer)?;
            route_id += 1;
        }
        writeln!(writer, "Cost {}", cost)?;
        writeln!(writer, "NB_ROUTES: {}", nb_routes)?;
        writeln!(writer, "CPU_TIME: {:.6}", cpu_time_secs)?;
        writer.flush()?;
        Ok(())
    }

    fn solve(
        data: Problem,
        params: Params,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<(Solution, i32, usize)>> {
        if params.decomp_nb_phases > 0 {
            if params.display_traces {
                println!("----- DECOMPOSITION ENABLED: ACTIVATING REVERSED MODE");
            }
            return reverse_mode::solve_reversed_mode(data, params, t0, save_solution);
        }

        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(data, params);
        Ok(ga.run(&mut rng, t0, save_solution, None).map(|(routes, cost)| {
            (Solution { routes: routes.clone() }, cost, routes.len())
        }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let t0 = Instant::now();
        let data = TigLoader::load(&challenge);
        let params = Params::initialize(hyperparameters, &data);
        match Self::solve(data, params, &t0, save_solution) {
            Ok(Some((solution, _cost, _routes))) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(e) => {
                eprintln!("Error: {}", e);
                Ok(None)
            }
        }
    }

    #[cfg(feature = "benchmark_io")]
    pub fn solve_benchmark_instance(
        run_type: &str,
        path: &str,
        hyperparameters: &Option<Map<String, Value>>,
        output_solution_path: Option<&str>,
        seed_byte: Option<u8>,
    ) -> Result<Option<Solution>> {
        let t0 = Instant::now();
        let mut data = match run_type {
            "vrptw" => VRPTWLoader::load(path)?,
            "cvrp" => CVRPLoader::load(path)?,
            other => return Err(anyhow!("Unknown benchmark run type '{}'", other)),
        };
        if let Some(seed) = seed_byte {
            data.seed = [seed; 32];
        }
        let params = Params::initialize(hyperparameters, &data);
        match Self::solve(data, params, &t0, None) {
            Ok(Some((solution, cost, routes))) => {
                if let Some(out_path) = output_solution_path {
                    let cpu_time_secs = t0.elapsed().as_secs_f64();
                    Self::write_solution_to_file(
                        out_path,
                        &solution,
                        cost,
                        routes,
                        cpu_time_secs,
                    )?;
                }
                Ok(Some(solution))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                eprintln!("Error: {}", e);
                Ok(None)
            }
        }
    }
}
