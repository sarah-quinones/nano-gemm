#![no_std]
#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex32;

include!(concat!(env!("OUT_DIR"), "/codegen.rs"));
