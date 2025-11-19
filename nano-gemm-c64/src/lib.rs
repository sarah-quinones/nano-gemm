#![no_std]
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex64;

include!(concat!(env!("OUT_DIR"), "/codegen.rs"));
