use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{to_string, to_value, Map, Value};
use std::{
    io::{Read, Write},
    str,
};

pub fn dejsonify<'a, T>(json_str: &'a str) -> serde_json::Result<T>
where
    T: Deserialize<'a>,
{
    serde_json::from_str::<T>(json_str)
}

pub fn jsonify<T>(obj: &T) -> String
where
    T: Serialize,
{
    to_string(&jsonify_internal(
        &to_value(obj).expect("to_value failed on serializable object"),
    ))
    .expect("to_string failed on serializable object")
}

pub fn jsonify_internal(json_value: &Value) -> Value {
    match json_value {
        Value::Object(obj) => {
            let mut sorted_map = Map::new();
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort();
            for key in keys {
                if let Some(value) = obj.get(key) {
                    sorted_map.insert(key.clone(), jsonify_internal(value));
                }
            }
            Value::Object(sorted_map)
        }
        _ => json_value.clone(),
    }
}

pub fn decompress_obj<T>(input: &[u8]) -> anyhow::Result<T>
where
    T: DeserializeOwned,
{
    let mut decoder = ZlibDecoder::new(input);
    let mut decompressed = String::new();
    decoder.read_to_string(&mut decompressed)?;
    Ok(dejsonify(&decompressed)?)
}

pub fn compress_obj<T>(input: T) -> Vec<u8>
where
    T: Serialize,
{
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(jsonify(&input).as_bytes()).unwrap();
    encoder.finish().unwrap()
}
