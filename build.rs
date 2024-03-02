// only targeting scalar, avx2, avx512
// no arm64 (until SVE comes), because no masked load instructions

// f32, f64, c32, c64

use std::fmt::Display;

struct RealKernel {
    ty: String,
    reg_ty: String,
    mask_ty: String,
    // register size
    n: usize,
    mr_div_n: usize,
    nr: usize,
    k: Option<usize>,

    target_features: String,
    set1: String,
    load_unaligned: String,
    store_unaligned: String,
    mask_load_unaligned: Box<dyn Fn(String, String) -> String>,

    mask_store_unaligned: String,
    mul_add: String,
    mul: String,
}

// f32 avx2 example, mr_div_n = 2, nr = 4, kr = Some(3)
#[allow(dead_code)]
mod example {
    // n = 8
    #[target_feature(enable = "avx,avx2,fma")]
    pub unsafe fn matmul_2_4_3(
        k: usize,
        dst: *mut f32,
        dst_cs: isize,
        lhs: *const f32,
        lhs_cs: isize,
        rhs: *const f32,
        rhs_rs: isize,
        rhs_cs: isize,
        beta: f32,
        last_mask: *const (),
    ) {
        _ = k;
        use core::arch::x86_64::*;
        type Reg = __m256;
        type Mask = __m256i;
        const N: isize = 8;

        let mut acc: [[Reg; 4]; 2] = core::mem::zeroed();
        let mut tmp_lhs: [Reg; 2] = core::mem::zeroed();

        let last_mask = *(last_mask as *const Mask);

        {
            let depth = 0;
            tmp_lhs[0] = _mm256_loadu_ps(lhs.offset(depth * lhs_cs + 0));
            tmp_lhs[1] = _mm256_maskload_ps(lhs.offset(depth * lhs_cs + N), last_mask);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 0 * rhs_cs));
            acc[0][0] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][0] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 1 * rhs_cs));
            acc[0][1] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][1] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 2 * rhs_cs));
            acc[0][2] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][2] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 3 * rhs_cs));
            acc[0][3] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][3] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);
        }
        {
            let depth = 1;
            tmp_lhs[0] = _mm256_loadu_ps(lhs.offset(depth * lhs_cs + 0));
            tmp_lhs[1] = _mm256_maskload_ps(lhs.offset(depth * lhs_cs + N), last_mask);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 0 * rhs_cs));
            acc[0][0] = _mm256_fmadd_ps(tmp_lhs[0], tmp_rhs, acc[0][0]);
            acc[1][0] = _mm256_fmadd_ps(tmp_lhs[1], tmp_rhs, acc[1][0]);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 1 * rhs_cs));
            acc[0][1] = _mm256_fmadd_ps(tmp_lhs[0], tmp_rhs, acc[0][0]);
            acc[1][1] = _mm256_fmadd_ps(tmp_lhs[1], tmp_rhs, acc[1][0]);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 2 * rhs_cs));
            acc[0][2] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][2] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 3 * rhs_cs));
            acc[0][3] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][3] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);
        }
        {
            let depth = 2;
            tmp_lhs[0] = _mm256_loadu_ps(lhs.offset(depth * lhs_cs + 0));
            tmp_lhs[1] = _mm256_maskload_ps(lhs.offset(depth * lhs_cs + N), last_mask);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 0 * rhs_cs));
            acc[0][0] = _mm256_fmadd_ps(tmp_lhs[0], tmp_rhs, acc[0][0]);
            acc[1][0] = _mm256_fmadd_ps(tmp_lhs[1], tmp_rhs, acc[1][0]);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 1 * rhs_cs));
            acc[0][1] = _mm256_fmadd_ps(tmp_lhs[0], tmp_rhs, acc[0][0]);
            acc[1][1] = _mm256_fmadd_ps(tmp_lhs[1], tmp_rhs, acc[1][0]);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 2 * rhs_cs));
            acc[0][2] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][2] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);

            let tmp_rhs = _mm256_set1_ps(*rhs.offset(depth * rhs_rs + 3 * rhs_cs));
            acc[0][3] = _mm256_mul_ps(tmp_lhs[0], tmp_rhs);
            acc[1][3] = _mm256_mul_ps(tmp_lhs[1], tmp_rhs);
        }

        let beta = _mm256_set1_ps(beta);

        {
            let dst = dst.offset(0 + 0 * dst_cs);
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(beta, acc[0][0], _mm256_loadu_ps(dst)));
        }
        {
            let dst = dst.offset(0 + 1 * dst_cs);
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(beta, acc[0][1], _mm256_loadu_ps(dst)));
        }
        {
            let dst = dst.offset(0 + 2 * dst_cs);
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(beta, acc[0][2], _mm256_loadu_ps(dst)));
        }
        {
            let dst = dst.offset(0 + 3 * dst_cs);
            _mm256_storeu_ps(dst, _mm256_fmadd_ps(beta, acc[0][3], _mm256_loadu_ps(dst)));
        }
        {
            let dst = dst.offset(N + 0 * dst_cs);
            _mm256_maskstore_ps(
                dst,
                last_mask,
                _mm256_fmadd_ps(beta, acc[1][0], _mm256_maskload_ps(dst, last_mask)),
            );
        }
        {
            let dst = dst.offset(N + 1 * dst_cs);
            _mm256_maskstore_ps(
                dst,
                last_mask,
                _mm256_fmadd_ps(beta, acc[1][1], _mm256_maskload_ps(dst, last_mask)),
            );
        }
        {
            let dst = dst.offset(N + 2 * dst_cs);
            _mm256_maskstore_ps(
                dst,
                last_mask,
                _mm256_fmadd_ps(beta, acc[1][2], _mm256_maskload_ps(dst, last_mask)),
            );
        }
        {
            let dst = dst.offset(N + 3 * dst_cs);
            _mm256_maskstore_ps(
                dst,
                last_mask,
                _mm256_fmadd_ps(beta, acc[1][3], _mm256_maskload_ps(dst, last_mask)),
            );
        }
    }
}

impl Display for RealKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // function that multiplies (mr_div_n * n, k) by (k, n_r)
        // C += beta * A * B

        // not exactly mr_div_n
        // the actual number of rows is between mr_div_n * (n-1) and mr_div_n * n
        write!(
            f,
            "#[target_feature(enable = \"{}\")]\n",
            self.target_features
        )?;
        write!(
            f,
            r#"pub unsafe fn matmul_{0:}_{1:}_{2:}(
                k: usize,
                dst: *mut {3:},
                dst_cs: isize,
                lhs: *const {3:},
                lhs_cs: isize,
                rhs: *const {3:},
                rhs_rs: isize,
                rhs_cs: isize,
                beta: {3:},
                last_mask: *const (),
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
            write!(f, "for depth in 0..k {{")?;
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

fn main() {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("codegen.rs");
    let kernel = RealKernel {
        ty: "f32".to_string(),
        reg_ty: "__m256".to_string(),
        mask_ty: "__m256i".to_string(),
        mr_div_n: 2,
        nr: 4,
        k: Some(3),
        target_features: "avx,avx2,fma".to_string(),
        n: 8,
        set1: "_mm256_set1_ps".to_string(),
        load_unaligned: "_mm256_loadu_ps".to_string(),
        store_unaligned: "_mm256_storeu_ps".to_string(),
        mask_load_unaligned: Box::new(|ptr, mask| format!("_mm256_maskload_ps({ptr}, {mask})")),
        mask_store_unaligned: "_mm256_maskstore_ps".to_string(),
        mul_add: "_mm256_fmadd_ps".to_string(),
        mul: "_mm256_mul_ps".to_string(),
    };
    std::fs::write(&dest_path, format!("{kernel}")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}
