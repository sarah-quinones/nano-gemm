use aligned_vec::avec;

extern crate intel_mkl_src;

const MIN: usize = 1;
const MAX: usize = 64;

#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn nanogemm(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {

    let a = avec![1.0; m.next_multiple_of(16) * k];
    let b = avec![1.0; k * n];
    let c = avec![0.0; m.next_multiple_of(16) * n];

    let beta = 2.5;

    let plan = nano_gemm::Plan::new_colmajor_lhs_and_dst_f32(m, n, k);
    let mut dst = c.clone();

    bencher.bench_local(|| unsafe {
        for _ in 0..1000 {
            plan.execute_unchecked(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                1,
                m.next_multiple_of(16) as isize,
                a.as_ptr(),
                1,
                m.next_multiple_of(16) as isize,
                b.as_ptr(),
                1,
                k as isize,
                1.0,
                beta,
                false,
                false,
            );
        }
    });
}



#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn nalgebra(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {

    let a = nalgebra::DMatrix::<f32>::zeros(m, k);
    let b = nalgebra::DMatrix::<f32>::zeros(k, n);
    let c = nalgebra::DMatrix::<f32>::zeros(m, n);

    let beta = 2.5;

    let mut dst = c.clone();

    bencher.bench_local(|| {
        for _ in 0..1000 {
            dst += beta * &a * &b
        }
    });
}


#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn faer(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {

    let a = faer::Mat::<f32>::zeros(m, k);
    let b = faer::Mat::<f32>::zeros(k, n);
    let c = faer::Mat::<f32>::zeros(m, n);

    let beta = 2.5;

    let mut dst = c.clone();

    bencher.bench_local(|| {
        for _ in 0..1000 {
            faer::modules::core::mul::matmul(
                dst.as_mut(),
                a.as_ref(),
                b.as_ref(),
                Some(1.0),
                beta,
                faer::Parallelism::None,
            )
        }
    });
}

#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn ndarray(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {

    let mut a = ndarray::Array2::zeros([m, k]);
    let mut b = ndarray::Array2::zeros([k, n]);
    let mut c = ndarray::Array2::zeros([m, n]);

    if false {
        a[(0, 0)] = 0.0f32;
        b[(0, 0)] = 0.0f32;
        c[(0, 0)] = 0.0f32;
    }

    let beta = 2.5f32;
    let mut dst = c.clone();
    bencher.bench_local(|| {
        for _ in 0..1000 {
            dst = &dst + beta * a.dot(&b)
        }
    });
}

pub fn main() {
    divan::main();
}
