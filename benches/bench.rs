use aligned_vec::avec;
use itertools::Itertools;

extern crate intel_mkl_src;

const MAX: usize = 128;

#[divan::bench(args = (1..=MAX).cartesian_product(1..=MAX).cartesian_product(1..=MAX).map(|((a, b), c)| [a, b, c]))]
pub fn nanogemm(bencher: divan::Bencher, [m, n, k]: [usize; 3]) {
    let a = avec![1.0; m.next_multiple_of(16) * k];
    let b = avec![1.0; k * n];
    let c = avec![0.0; m.next_multiple_of(16) * n];

    let beta = 2.5;

    let plan = nano_gemm::PlanReal::new_colmajor_lhs_and_dst_f32(m, n, k);
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
            );
        }
    });
}

#[divan::bench(args = (1..=MAX).cartesian_product(1..=MAX).cartesian_product(1..=MAX).map(|((a, b), c)| [a, b, c]))]
pub fn nalgebra(bencher: divan::Bencher, [m, n, k]: [usize; 3]) {
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

#[divan::bench(args = (1..=MAX).cartesian_product(1..=MAX).cartesian_product(1..=MAX).map(|((a, b), c)| [a, b, c]))]
pub fn ndarray(bencher: divan::Bencher, [m, n, k]: [usize; 3]) {
    if std::is_x86_feature_detected!("avx")
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
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
}

pub fn main() {
    divan::main();
}
