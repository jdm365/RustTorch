fn main() {
    println!("Training MNIST on GPU");
    mlp::run("data/").map_err(|err| println!("{:?}", err)).unwrap();
}
