// only targeting scalar, avx2, avx512
// no arm64 (until SVE comes), because no masked load instructions

// f32, f64, c32, c64

use std::fmt::Display;
use std::fmt::Write;

type Str = &'static str;

struct RealKernel {
    ty: Str,
    reg_ty: Str,
    mask_ty: Str,
    // register size
    n: usize,
    mr_div_n: usize,
    nr: usize,
    k: Option<usize>,

    target_features: Str,
    set1: Str,
    load_unaligned: Str,
    store_unaligned: Str,
    mask_load_unaligned: Box<dyn Fn(String, String) -> String>,

    mask_store_unaligned: Str,
    mul_add: Str,
    mul: Str,
}

impl Display for RealKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // function that multiplies (mr_div_n * n, k) by (k, n_r)
        // C += beta * A * B

        // not exactly mr_div_n
        // the actual number of rows is between mr_div_n * (n-1) and mr_div_n * n
        write!(
            f,
            "
            #[target_feature(enable = \"{}\")]\n",
            self.target_features
        )?;
        write!(
            f,
            r#"pub unsafe fn matmul_{0:}_{1:}_{2:}(
                &crate::MicroKernelData {{ alpha, beta, k, dst_cs, lhs_cs, rhs_rs, rhs_cs, last_mask }}: &crate::MicroKernelData< {3:} >,
                dst: *mut {3:},
                lhs: *const {3:},
                rhs: *const {3:},
            ) {{
"#,
            self.mr_div_n,
            self.nr,
            self.k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
            self.ty,
        )?;
        write!(
            f,
            r#"
                #[cfg(target_arch = "x86_64")]
                use core::arch::x86_64::*;
                #[cfg(target_arch = "x86")]
                use core::arch::x86::*;
            "#
        )?;

        write!(f, "_ = k;\n")?;
        write!(f, "type Reg = {};\n", self.reg_ty)?;
        write!(f, "type Mask = {};\n", self.mask_ty)?;
        write!(f, "const N: isize = {};\n", self.n)?;
        write!(
            f,
            "let mut acc: [[Reg; {}]; {}] = core::mem::zeroed();\n",
            self.nr, self.mr_div_n
        )?;
        write!(
            f,
            "let mut tmp_lhs: [Reg; {}] = core::mem::zeroed();\n",
            self.mr_div_n
        )?;

        write!(f, "let last_mask = *(last_mask as *const Mask);\n")?;

        if let Some(k) = self.k {
            for depth in 0..k {
                write!(f, "let depth = {depth};\n")?;
                for i in 0..self.mr_div_n {
                    self.write_load_lhs(i, f)?;
                }
                for j in 0..self.nr {
                    write!(
                        f,
                        "let tmp_rhs = {}(*rhs.offset(depth * rhs_rs + {j} * rhs_cs));\n",
                        self.set1
                    )?;

                    for i in 0..self.mr_div_n {
                        if depth > 0 {
                            write!(
                                f,
                                "acc[{i}][{j}] = {}(tmp_lhs[{i}], tmp_rhs, acc[{i}][{j}]);\n",
                                self.mul_add
                            )?;
                        } else {
                            write!(f, "acc[{i}][{j}] = {}(tmp_lhs[{i}], tmp_rhs);\n", self.mul)?;
                        }
                    }
                }
            }
        } else {
            write!(f, "for depth in 0..k as isize {{")?;
            for i in 0..self.mr_div_n {
                self.write_load_lhs(i, f)?;
            }
            for j in 0..self.nr {
                write!(
                    f,
                    "let tmp_rhs = {}(*rhs.offset(depth * rhs_rs + {j} * rhs_cs));\n",
                    self.set1
                )?;

                for i in 0..self.mr_div_n {
                    write!(
                        f,
                        "acc[{i}][{j}] = {}(tmp_lhs[{i}], tmp_rhs, acc[{i}][{j}]);\n",
                        self.mul_add
                    )?;
                }
            }
            write!(f, "}}")?;
        }

        write!(f, "if alpha == 1.0 {{")?;
        write!(f, "let beta = {}(beta);\n", self.set1)?;
        for j in 0..self.nr {
            for i in 0..self.mr_div_n {
                write!(f, "{{")?;
                write!(f, "let dst = dst.offset({i} * N + {j} * dst_cs);")?;
                if i + 1 < self.mr_div_n {
                    write!(
                        f,
                        "{}(dst, {}(beta, acc[{i}][{j}], {}(dst)));\n",
                        self.store_unaligned, self.mul_add, self.load_unaligned
                    )?;
                } else {
                    write!(
                        f,
                        "{}(dst, last_mask, {}(beta, acc[{i}][{j}], {}));\n",
                        self.mask_store_unaligned,
                        self.mul_add,
                        (self.mask_load_unaligned)(format!("dst"), "last_mask".to_string()),
                    )?;
                }
                write!(f, "}}")?;
            }
        }
        write!(f, "}}")?;
        write!(f, "else if alpha == 0.0 {{")?;
        write!(f, "let beta = {}(beta);\n", self.set1)?;
        for j in 0..self.nr {
            for i in 0..self.mr_div_n {
                write!(f, "{{")?;
                write!(f, "let dst = dst.offset({i} * N + {j} * dst_cs);")?;
                if i + 1 < self.mr_div_n {
                    write!(
                        f,
                        "{}(dst, {}(beta, acc[{i}][{j}]));\n",
                        self.store_unaligned, self.mul
                    )?;
                } else {
                    write!(
                        f,
                        "{}(dst, last_mask, {}(beta, acc[{i}][{j}]));\n",
                        self.mask_store_unaligned, self.mul,
                    )?;
                }
                write!(f, "}}")?;
            }
        }
        write!(f, "}}")?;
        write!(f, "else {{")?;
        write!(f, "let beta = {}(beta);\n", self.set1)?;
        write!(f, "let alpha = {}(alpha);\n", self.set1)?;
        for j in 0..self.nr {
            for i in 0..self.mr_div_n {
                write!(f, "{{")?;
                write!(f, "let dst = dst.offset({i} * N + {j} * dst_cs);")?;
                if i + 1 < self.mr_div_n {
                    write!(
                        f,
                        "{}(dst, {}(beta, acc[{i}][{j}], {}({}(dst), alpha)));\n",
                        self.store_unaligned, self.mul_add, self.mul, self.load_unaligned
                    )?;
                } else {
                    write!(
                        f,
                        "{}(dst, last_mask, {}(beta, acc[{i}][{j}], {}({}, alpha)));\n",
                        self.mask_store_unaligned,
                        self.mul_add,
                        self.mul,
                        (self.mask_load_unaligned)(format!("dst"), "last_mask".to_string()),
                    )?;
                }
                write!(f, "}}")?;
            }
        }
        write!(f, "}}")?;

        write!(f, "}}\n")
    }
}

impl RealKernel {
    fn write_load_lhs(
        &self,
        i: usize,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        Ok(if i + 1 < self.mr_div_n {
            write!(
                f,
                "tmp_lhs[{i}] = {}(lhs.offset(depth * lhs_cs + {i} * N));\n",
                self.load_unaligned
            )?;
        } else {
            write!(
                f,
                "tmp_lhs[{i}] = {};\n",
                (self.mask_load_unaligned)(
                    format!("lhs.offset(depth * lhs_cs + {i} * N)"),
                    "last_mask".to_string()
                ),
            )?;
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("codegen.rs");

    let mut code = String::new();

    write!(
        code,
        r#"
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;
        "#
    )?;
    write!(
        code,
        "
        #[cfg(any(target_arch = \"x86\", target_arch = \"x86_64\"))]
        pub mod avx {{\n"
    )?;
    write!(code, "pub mod f32 {{\n")?;
    for mr_div_n in 1..=2 {
        for nr in 1..=4 {
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                let kernel = RealKernel {
                    ty: "f32",
                    reg_ty: "__m256",
                    mask_ty: "__m256i",
                    mr_div_n,
                    nr,
                    k,
                    target_features: "avx,avx2,fma",
                    n: 8,
                    set1: "_mm256_set1_ps",
                    load_unaligned: "_mm256_loadu_ps",
                    store_unaligned: "_mm256_storeu_ps",
                    mask_load_unaligned: Box::new(|ptr, mask| {
                        format!("_mm256_maskload_ps({ptr}, {mask})")
                    }),
                    mask_store_unaligned: "_mm256_maskstore_ps",
                    mul_add: "_mm256_fmadd_ps",
                    mul: "_mm256_mul_ps",
                };

                write!(code, "{kernel}")?;
            }
        }
    }

    write!(
        code,
        "pub static MICROKERNELS: [[[crate::MicroKernel<f32>; 4]; 2]; 17] = [\n"
    )?;
    for k in (1..=16).into_iter().map(Some).chain([None]) {
        write!(code, "[\n")?;
        for mr_div_n in 1..=2 {
            write!(code, "[\n")?;
            for nr in 1..=4 {
                write!(
                    code,
                    "matmul_{mr_div_n}_{nr}_{},",
                    k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
                )?;
            }
            write!(code, "],\n")?;
        }
        write!(code, "],\n")?;
    }
    write!(code, "];\n")?;
    write!(
        code,
        "
            pub static MASKS: [crate::__m256i; 8] = unsafe {{ core::mem::transmute([
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],

                [u32::MAX, 0, 0, 0, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, 0, 0, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, 0, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0],
            ]) }};
        "
    )?;

    write!(code, "}}\n")?;
    write!(code, "}}\n")?;

    write!(
        code,
        "
        #[cfg(any(target_arch = \"x86\", target_arch = \"x86_64\"))]
        #[cfg(feature = \"nightly\")]
        pub mod avx512 {{\n"
    )?;
    write!(code, "pub mod f32 {{\n")?;
    for mr_div_n in 1..=2 {
        for nr in 1..=4 {
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                let kernel = RealKernel {
                    ty: "f32",
                    reg_ty: "__m512",
                    mask_ty: "u16",
                    mr_div_n,
                    nr,
                    k,
                    target_features: "avx512f",
                    n: 16,
                    set1: "_mm512_set1_ps",
                    load_unaligned: "_mm512_loadu_ps",
                    store_unaligned: "_mm512_storeu_ps",
                    mask_load_unaligned: Box::new(|ptr, mask| {
                        format!("_mm512_maskz_loadu_ps({mask}, {ptr})")
                    }),
                    mask_store_unaligned: "_mm512_mask_storeu_ps",
                    mul_add: "_mm512_fmadd_ps",
                    mul: "_mm512_mul_ps",
                };

                write!(code, "{kernel}")?;
            }
        }
    }

    write!(
        code,
        "pub static MICROKERNELS: [[[crate::MicroKernel<f32>; 4]; 2]; 17] = [\n"
    )?;
    for k in (1..=16).into_iter().map(Some).chain([None]) {
        write!(code, "[\n")?;
        for mr_div_n in 1..=2 {
            write!(code, "[\n")?;
            for nr in 1..=4 {
                write!(
                    code,
                    "matmul_{mr_div_n}_{nr}_{},",
                    k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
                )?;
            }
            write!(code, "],\n")?;
        }
        write!(code, "],\n")?;
    }
    write!(code, "];\n")?;
    write!(
        code,
        "
            pub static MASKS: [u16; 16] = [
                0b1111_1111_1111_1111,
                0b0000_0000_0000_0001,
                0b0000_0000_0000_0011,
                0b0000_0000_0000_0111,
                0b0000_0000_0000_1111,
                0b0000_0000_0001_1111,
                0b0000_0000_0011_1111,
                0b0000_0000_0111_1111,
                0b0000_0000_1111_1111,
                0b0000_0001_1111_1111,
                0b0000_0011_1111_1111,
                0b0000_0111_1111_1111,
                0b0000_1111_1111_1111,
                0b0001_1111_1111_1111,
                0b0011_1111_1111_1111,
                0b0111_1111_1111_1111,
            ];
        "
    )?;

    write!(code, "}}\n")?;
    write!(code, "}}\n")?;

    std::fs::write(&dest_path, format!("{code}")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
