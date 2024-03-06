fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("codegen.rs");

    let mut code = String::new();

    code.push_str(
        r#"
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub mod x86 {
        "#,
    );
    code.push_str(
        r###"
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[target_feature(enable = "avx512f")]
#[inline]
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "nightly"))]
unsafe fn subadd_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    _mm512_fmaddsub_pd(
        a,
        b,
        core::mem::transmute(_mm512_xor_si512(
            core::mem::transmute(c),
            core::mem::transmute(_mm512_set1_pd(-0.0)),
        )),
    )
}
"###,
    );
    code.push_str(&nano_gemm_codegen::x86::codegen_c64()?);
    code.push_str("}");

    code.push_str(
        r#"
        #[cfg(target_arch = "aarch64")]
        pub mod aarch64 {
    "#,
    );
    code.push_str(&nano_gemm_codegen::aarch64::codegen_c64()?);
    code.push_str("}");

    std::fs::write(&dest_path, format!("{code}")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
