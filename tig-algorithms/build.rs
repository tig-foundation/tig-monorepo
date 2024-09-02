use std::env;
use std::fs;
use std::path::Path;
use regex::Regex;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();


    let challenges = vec!["knapsack", "vehicle_routing", "satisfiability", "vector_search"];
   

    let re_export_regex = Regex::new(r"pub use (?P<name>\w+) as (?P<alias>\w+);").unwrap();
    for challenge in challenges {
        let mut solver_code = String::new();
        let mut algorithms = String::new();
        let mut existing_algos = String::new();
        let mut existing_algos_names = String::new();

        let dest_path = Path::new(&out_dir).join(format!("{challenge}_solver.rs"));
        let mod_file_path = Path::new("src").join(challenge).join("mod.rs");
        if mod_file_path.exists() {
            
            let mod_file_content = fs::read_to_string(mod_file_path).unwrap();
            for cap in re_export_regex.captures_iter(&mod_file_content) {
                let name = &cap["name"];
                let alias = &cap["alias"];
                let function_name = format!("{}::solve_challenge", name);
                algorithms.push_str(&format!(
                    "\"{}\" => Some({}),\n",
                    alias, function_name
                ));
                existing_algos.push_str(&format!("\"{}\" => true,\n", alias));
                existing_algos_names.push_str(&format!("\"{}\" => true,\n", name));
            }
            
            let solver_template = fs::read_to_string("src/solver_template.rs")
            .expect("Failed to read src/entry_point_template.rs");
            // Generate the code with the substituted values
            solver_code = solver_template
                .replace("{CHALLENGE}", challenge)
                .replace("{ALGORITHMS}", &algorithms)
                .replace("{EXISTING_ALGOS}", &existing_algos)
                .replace("{EXISTING_ALGOS_NAMES}", &existing_algos_names);
        }
        fs::write(dest_path, solver_code).unwrap();
    }

    println!("cargo:rerun-if-changed=src");
}