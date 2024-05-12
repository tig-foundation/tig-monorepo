import init, { setup, state, start, stop, select_algorithm } from './pkg/tig_benchmarker.js';

async function loadWasm() {
  console.log("Loading Benchmarker WASM");
  await init("./pkg/tig_benchmarker_bg.wasm");
  window.benchmarker = {
    setup, state, start, stop, select_algorithm
  };
  console.log("Benchmarker WASM initialized and functions are available globally");
}

loadWasm();