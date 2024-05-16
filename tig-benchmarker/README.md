# tig-benchmarker

A Rust crate that implements a Benchmarker for TIG that can run in the browser. 

This browser benchmarker is deployed to https://play.tig.foundation/benchmarker

To run it locally, run the following commands before visiting localhost in your browser:

```
# uncomment below to install wasm-pack
# curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
cd tig-benchmarker
wasm-pack build --release --target web
python3 -m http.server 80
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)