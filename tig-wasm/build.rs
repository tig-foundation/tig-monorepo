use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=CHALLENGE");
    println!("cargo:rerun-if-env-changed=ALGORITHM");
    // Only run the following code if the "entry-point" feature is enabled
    if env::var("CARGO_FEATURE_ENTRY_POINT").is_ok() {
        // Read the CHALLENGE and ALGORITHM environment variables
        let challenge = env::var("CHALLENGE").expect("CHALLENGE environment variable not set");
        let algorithm = env::var("ALGORITHM").expect("ALGORITHM environment variable not set");

        let entry_point_template = fs::read_to_string("src/entry_point_template.rs")
            .expect("Failed to read src/entry_point_template.rs");
        // Generate the code with the substituted values
        let code = entry_point_template
            .replace("{CHALLENGE}", challenge.as_str())
            .replace("{ALGORITHM}", algorithm.as_str());

        // Write the generated code to a file
        let out_dir = env::var("OUT_DIR").unwrap();
        let dest_path = Path::new(&out_dir).join("entry_point.rs");
        fs::write(dest_path, code).expect("Failed to write entry_point.rs");
    }
}
