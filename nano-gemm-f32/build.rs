fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("codegen.rs");

    let mut code = String::new();

    code.push_str(
        r#"
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub mod x86 {
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;

            #[inline(always)]
            unsafe fn splat_1s(x: f32) -> __m128 {
                _mm_load_ss(&x)
            }

            #[inline(always)]
            unsafe fn load_2s(ptr: *const f32) -> __m128 {
                core::mem::transmute(_mm_load_sd(ptr as _))
            }

            #[inline(always)]
            unsafe fn store_2s(ptr: *mut f32, v: __m128) {
                _mm_store_sd(ptr as _, core::mem::transmute(v))
            }

        "#,
    );
    code.push_str(&nano_gemm_codegen::x86::codegen_f32()?);
    code.push_str("}");

    code.push_str(
        r#"
        #[cfg(target_arch = "aarch64")]
        pub mod aarch64 {
    "#,
    );
    code.push_str(&nano_gemm_codegen::aarch64::codegen_f32()?);
    code.push_str("}");

    std::fs::write(&dest_path, format!("{code}")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
