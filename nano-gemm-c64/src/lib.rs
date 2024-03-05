#![cfg_attr(
    all(any(target_arch = "x86_64", target_arch = "x86"), feature = "nightly"),
    feature(stdarch_x86_avx512, avx512_target_feature)
)]
#![no_std]
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex64;

include!(concat!(env!("OUT_DIR"), "/codegen.rs"));
