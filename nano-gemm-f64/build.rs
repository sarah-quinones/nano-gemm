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
        unsafe fn splat_1d(x: f64) -> __m128d {
            _mm_load_sd(&x)
        }
        "#,
    );
    code.push_str(&nano_gemm_codegen::x86::codegen_f64()?);
    code.push_str("}");

    code.push_str(
        r#"
        #[cfg(target_arch = "aarch64")]
        pub mod aarch64 {
    "#,
    );
    code.push_str(&nano_gemm_codegen::aarch64::codegen_f64()?);
    code.push_str("}");

    std::fs::write(&dest_path, format!("{code}")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
