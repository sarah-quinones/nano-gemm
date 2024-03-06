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
unsafe fn subadd_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    _mm512_fmaddsub_ps(
        a,
        b,
        core::mem::transmute(_mm512_xor_si512(
            core::mem::transmute(c),
            core::mem::transmute(_mm512_set1_ps(-0.0)),
        )),
    )
}

#[inline(always)]
unsafe fn load_2s(ptr: *const f32) -> __m128 {
    core::mem::transmute(_mm_load_sd(ptr as _))
}

#[inline(always)]
unsafe fn store_2s(ptr: *mut f32, v: __m128) {
    _mm_store_sd(ptr as _, core::mem::transmute(v))
}
"###,
    );
    code.push_str(&nano_gemm_codegen::x86::codegen_c32()?);
    code.push_str("}");

    code.push_str(
        r#"
        #[cfg(target_arch = "aarch64")]
        pub mod aarch64 {
    "#,
    );
    code.push_str(&nano_gemm_codegen::aarch64::codegen_c32()?);
    code.push_str("}");

    std::fs::write(&dest_path, format!("{code}")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
