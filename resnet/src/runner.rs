use anyhow::Result;
use tch::{nn, Device, nn::ModuleT, nn::OptimizerConfig};

use crate::resnet::resnet18;

const BATCH_SIZE: i64 = 256;
const N_EPOCHS: i64 = 10;
const LR: f64 = 1e-2;

pub fn run(data: &str) -> Result<()> {
    // default to data if not provided
    let data = if data.is_empty() { "data" } else { data };
    let m = tch::vision::mnist::load_dir(data)?;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let model = resnet18(&vs.root(), 10);
    let mut opt = nn::Adam::default().build(&vs, LR)?;
    for epoch in 0..N_EPOCHS {
        for (bimages, blabels) in m.train_iter(BATCH_SIZE).shuffle().to_device(vs.device()) {
            let _bimages = bimages.view([-1, 1, 28, 28]);

            let loss = model.forward_t(&_bimages, true).cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let _test_images = m.test_images.view([-1, 1, 28, 28]);
        let test_accuracy = model.batch_accuracy_for_logits(
                                    &_test_images,
                                    &m.test_labels, 
                                    vs.device(), 
                                    BATCH_SIZE
                                    );
        println!("epoch: {:4} test acc: {:5.2}%", epoch + 1, 100. * test_accuracy);
    }
    Ok(())
}
