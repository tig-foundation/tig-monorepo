use {
    libloading::{Library},
    std::path::Path,
    std::panic,
};

pub fn load_module(path: &Path) -> Result<Library, String> 
{
    let res = panic::catch_unwind(|| {
        unsafe { Library::new(path) }
    });

    match res {
        Ok(lib_result) => lib_result.map_err(|e| e.to_string()),
        Err(_) => Err("Failed to load module".to_string())
    }
}