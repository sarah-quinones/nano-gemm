use aligned_vec::avec;

const MIN: usize = 1;
const MAX: usize = 64;

type T = f32;
type FaerT = f32;
const ONE: T = 1.0;
const BETA: T = 2.5;
const ALPHA: T = 3.7;

#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [size, size, 4]))
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn nanogemm_plan(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {
    let stride = m.next_multiple_of(Ord::min(16, m.next_power_of_two()));

    let a = avec![ONE; stride * k];
    let b = avec![ONE; k * n];
    let c = avec![ONE; stride * n];

    let beta = BETA;
    let alpha = ALPHA;

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
                stride as isize,
                a.as_ptr(),
                1,
                stride as isize,
                b.as_ptr(),
                1,
                k as isize,
                alpha,
                beta.into(),
                false,
                false,
            );
        }
    });
}

#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [size, size, 4]))
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn nanogemm_noplan(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {
    let stride = m.next_multiple_of(Ord::min(16, m.next_power_of_two()));

    let a = avec![ONE; stride * k];
    let b = avec![ONE; k * n];
    let c = avec![ONE; stride * n];

    let beta = BETA;
    let alpha = ALPHA;

    let mut dst = c.clone();

    bencher.bench_local(|| unsafe {
        for _ in 0..1000 {
            nano_gemm::planless::execute_f32(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                n as isize,
                core::hint::black_box(1),
                a.as_ptr(),
                k as isize,
                core::hint::black_box(1),
                b.as_ptr(),
                n as isize,
                1,
                alpha,
                beta.into(),
                false,
                false,
            );
        }
    });
}

#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [size, size, 4]))
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn nanogemm_noplan_suboptimal(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {
    let stride = m.next_multiple_of(Ord::min(16, m.next_power_of_two()));

    let a = avec![ONE; stride * k];
    let b = avec![ONE; k * n];
    let c = avec![ONE; stride * n];

    let beta = BETA;
    let alpha = ALPHA;

    let mut dst = c.clone();

    bencher.bench_local(|| unsafe {
        for _ in 0..1000 {
            nano_gemm::planless::execute_f32(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                k as isize,
                core::hint::black_box(1),
                a.as_ptr(),
                n as isize,
                core::hint::black_box(1),
                b.as_ptr(),
                1,
                k as isize,
                alpha,
                beta.into(),
                false,
                false,
            );
        }
    });
}

#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [size, size, 4]))
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn nalgebra(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {

    let a = nalgebra::DMatrix::<T>::zeros(m, k);
    let b = nalgebra::DMatrix::<T>::zeros(k, n);
    let c = nalgebra::DMatrix::<T>::zeros(m, n);

    let mut dst = c.clone();

    bencher.bench_local(|| {
        for _ in 0..1000 {
            a.mul_to(&b, &mut dst);
        }
    });
}

#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [size, size, 4]))
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn faer(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {

    let a = faer::Mat::<FaerT>::zeros(m, k);
    let b = faer::Mat::<FaerT>::zeros(k, n);
    let c = faer::Mat::<FaerT>::zeros(m, n);

    let beta = 2.5.into();

    let mut dst = c.clone();

    bencher.bench_local(|| {
        for _ in 0..1000 {
            faer::modules::core::mul::matmul(
                dst.as_mut(),
                a.as_ref(),
                b.as_ref(),
                Some(1.0.into()),
                beta,
                faer::Parallelism::None,
            )
        }
    });
}

#[cfg(target_arch = "x86_64")]
extern crate intel_mkl_src;

#[cfg(target_arch = "x86_64")]
#[divan::bench(args = 
    (MIN..=MAX).map(|size| [size, size, size])
    .chain((MIN..=MAX).map(|size| [size, size, 4]))
    .chain((MIN..=MAX).map(|size| [4, size, 4]))
    .chain((MIN..=MAX).map(|size| [size, 4, 4]))
)]
pub fn ndarray(bencher: divan::Bencher, [ m, n, k]: [usize; 3]) {

    let mut a = ndarray::Array2::zeros([m, k]);
    let mut b = ndarray::Array2::zeros([k, n]);
    let mut c = ndarray::Array2::zeros([m, n]);

    let beta = BETA;

    if false {
        a[(0, 0)] = beta;
        b[(0, 0)] = beta;
        c[(0, 0)] = beta;
    }

    let dst = c.clone();
    bencher.bench_local(|| {
        for _ in 0..1000 {
             _ = &dst + beta * a.dot(&b);
        }
    });
}

pub fn main() {
    divan::main();
}
