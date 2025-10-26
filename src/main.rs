fn main() {
    let raw_image = rawloader::decode_file("IMG_20251023_132747.dng").unwrap();
    let raw_data = match raw_image.data {
        rawloader::RawImageData::Integer(data) => data,
        _ => panic!("Non-integer raw file processing is not supported"),
    };

    println!("{:?}", raw_data);
}
