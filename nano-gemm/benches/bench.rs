use aligned_vec::avec;
use diol::prelude::*;

const MIN: usize = 1;
const MAX: usize = 128;

type T = f32;
type FaerT = f32;
const ONE: T = 1.0;
const BETA: T = 2.5;
const ALPHA: T = 3.7;

fn args() -> Vec<[usize; 3]> {
    (MIN..=MAX)
        .map(|size| [size, size, size])
        .chain((MIN..=MAX).map(|size| [size, size, 4]))
        .chain((MIN..=MAX).map(|size| [4, size, 4]))
        .chain((MIN..=MAX).map(|size| [size, 4, 4]))
        .collect()
}

pub fn nanogemm_plan(bencher: Bencher, [m, n, k]: [usize; 3]) {
    let stride = m.next_multiple_of(Ord::min(16, m.next_power_of_two()));

    let a = avec![ONE; stride * k];
    let b = avec![ONE; k * n];
    let c = avec![ONE; stride * n];

    let beta = BETA;
    let alpha = ALPHA;

    let plan = nano_gemm::Plan::new_colmajor_lhs_and_dst_f32(m, n, k);
    let mut dst = c.clone();

    bencher.bench(|| unsafe {
        {
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

pub fn nanogemm_noplan(bencher: Bencher, [m, n, k]: [usize; 3]) {
    let stride = m.next_multiple_of(Ord::min(16, m.next_power_of_two()));

    let a = avec![ONE; stride * k];
    let b = avec![ONE; k * n];
    let c = avec![ONE; stride * n];

    let beta = BETA;
    let alpha = ALPHA;

    let mut dst = c.clone();

    bencher.bench(|| unsafe {
        {
            nano_gemm::planless::execute_f32(
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

pub fn nanogemm_noplan_suboptimal(bencher: Bencher, [m, n, k]: [usize; 3]) {
    let stride = m.next_multiple_of(Ord::min(16, m.next_power_of_two()));

    let a = avec![ONE; stride * k];
    let b = avec![ONE; k * n];
    let c = avec![ONE; stride * n];

    let beta = BETA;
    let alpha = ALPHA;

    let mut dst = c.clone();

    bencher.bench(|| unsafe {
        {
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

pub fn nalgebra(bencher: Bencher, [m, n, k]: [usize; 3]) {
    let a = nalgebra::DMatrix::<T>::zeros(m, k);
    let b = nalgebra::DMatrix::<T>::zeros(k, n);
    let c = nalgebra::DMatrix::<T>::zeros(m, n);

    let mut dst = c.clone();

    bencher.bench(|| {
        a.mul_to(&b, &mut dst);
    });
}

pub fn faer(bencher: Bencher, [m, n, k]: [usize; 3]) {
    let a = faer::Mat::<FaerT>::zeros(m, k);
    let b = faer::Mat::<FaerT>::zeros(k, n);
    let c = faer::Mat::<FaerT>::zeros(m, n);

    let beta = 2.5.into();

    let mut dst = c.clone();

    bencher.bench(|| {
        {
            faer::modules::core::mul::matmul(
                dst.as_mut(),
                a.as_ref(),
                b.as_ref(),
                Some(1.0.into()),
                beta,
                faer::Parallelism::Rayon(0),
            )
        }
    });
}

#[cfg(target_arch = "x86_64")]
extern crate intel_mkl_src;

#[cfg(target_arch = "x86_64")]
pub fn ndarray(bencher: Bencher, [m, n, k]: [usize; 3]) {
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
    bencher.bench(|| {
        _ = &dst + beta * a.dot(&b);
    });
}

pub fn main() {
    let mut config = BenchConfig::default();

    use clap::Parser;
    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long)]
        bench: bool,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        arg: Option<String>,
    }
    let cmdline_args = Args::parse();
    if let Some(name) = &cmdline_args.name {
        config.fn_regex = Some(Regex::new(name).unwrap());
    }
    if let Some(arg) = &cmdline_args.arg {
        config.arg_regex = Some(Regex::new(arg).unwrap());
    }
    config.split = Split::ByArg;

    let mut bench = Bench::new(config);

    #[cfg(target_arch = "x86_64")]
    bench.register_many(
        list![
            nanogemm_plan,
            nanogemm_noplan,
            nanogemm_noplan_suboptimal,
            nalgebra,
            faer,
            ndarray,
        ],
        args(),
    );
    #[cfg(not(target_arch = "x86_64"))]
    bench.register_many(
        list![
            nanogemm_plan,
            nanogemm_noplan,
            nanogemm_noplan_suboptimal,
            nalgebra,
            faer,
        ],
        args(),
    );

    bench.run();
}
