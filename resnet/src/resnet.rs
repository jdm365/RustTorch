use tch::nn;

fn conv2d(
    vs: &nn::Path, 
    c_in: i64, 
    c_out: i64, 
    kernel_size: i64, 
    stride: i64, 
    padding: i64
    ) -> nn::Conv2D {

    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding: padding,
        bias: false,
        ..Default::default()
    };
    return nn::conv2d(vs, c_in, c_out, kernel_size, conv2d_cfg);
}



fn downsample(
    vs: &nn::Path, 
    c_in: i64, 
    c_out: i64, 
    stride: i64
    ) -> impl nn::ModuleT {

    if stride != 1 || c_in != c_out {
        return nn::seq_t()
            .add(conv2d(vs, c_in, c_out, 1, stride, 0))
            .add(nn::batch_norm2d(vs, c_out, Default::default()));
    } 
    else {
        return nn::seq_t();
    }
}



fn residual_block(
    vs: &nn::Path, 
    c_in: i64, 
    c_out: i64, 
    stride: i64
    ) -> impl nn::ModuleT {

    let conv1 = conv2d(vs, c_in, c_out, 3, stride, 1);
    let bn1   = nn::batch_norm2d(vs, c_out, Default::default());
    let conv2 = conv2d(vs, c_out, c_out, 3, 1, 1);
    let bn2   = nn::batch_norm2d(vs, c_out, Default::default());
    let downsample = downsample(vs, c_in, c_out, stride);

    return nn::func_t(move |xs, train| {
        let ys = xs.apply(&conv1)
                   .apply_t(&bn1, train)
                   .relu()
                   .apply(&conv2)
                   .apply_t(&bn2, train);
        return (xs.apply_t(&downsample, train) + ys).relu();
        }
    )
}


fn stack_residual_blocks(
    vs: &nn::Path, 
    c_in: i64, 
    c_out: i64, 
    stride: i64, 
    num_blocks: i64
    ) -> impl nn::ModuleT {

    let mut blocks = nn::seq_t().add(residual_block(vs, c_in, c_out, stride));
    for _ in 1..num_blocks {
        blocks = blocks.add(residual_block(vs, c_out, c_out, 1));
    }
    return blocks;
}


fn resnet(
    vs: &nn::Path,
    mod1_blocks: i64,
    mod2_blocks: i64,
    mod3_blocks: i64,
    mod4_blocks: i64,
    num_classes: i64
    ) -> nn::FuncT<'static> {
    let init_conv = conv2d(vs, 1, 64, 3, 1, 1);
    let init_bn   = nn::batch_norm2d(vs, 64, Default::default());

    let block1 = stack_residual_blocks(vs, 64, 64, 1, mod1_blocks);
    let block2 = stack_residual_blocks(vs, 64, 128, 2, mod2_blocks);
    let block3 = stack_residual_blocks(vs, 128, 256, 2, mod3_blocks);
    let block4 = stack_residual_blocks(vs, 256, 512, 2, mod4_blocks);

    let fc = nn::linear(vs, 512, num_classes, Default::default());

    return nn::func_t(move |xs, train| {
        return xs.apply(&init_conv)
                   .apply_t(&init_bn, train)
                   .relu()
                   .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
                   .apply_t(&block1, train)
                   .apply_t(&block2, train)
                   .apply_t(&block3, train)
                   .apply_t(&block4, train)
                   .adaptive_avg_pool2d(&[1, 1])
                   .view([-1, 512])
                   //.apply_opt(&fc);
                   .apply_t(&fc, train);
    });
}


pub fn resnet18(vs: &nn::Path, num_classes: i64) -> nn::FuncT<'static> {
    return resnet(vs, 2, 2, 2, 2, num_classes);
}
