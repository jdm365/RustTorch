mod runner;
mod resnet;


fn main() {
    println!("");
    println!("==============================");
    println!("=== Training ResNet on GPU ===");
    println!("==============================");

    runner::run("data/").map_err(|err| println!("{:?}", err)).unwrap();
}
