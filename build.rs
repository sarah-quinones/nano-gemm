// only targeting scalar, avx2, avx512
// no arm64 (until SVE comes), because no masked load instructions

// f32, f64, c32, c64

use std::fmt::Display;
use std::fmt::Write;

struct RealKernel {
    ty: &'static str,
    reg_ty: &'static str,
    mask_ty: &'static str,
    // register size
    n: usize,
    mr_div_n: usize,
    nr: usize,
    k: Option<usize>,

    target_features: &'static str,
    set1: &'static str,
    load_unaligned: &'static str,
    store_unaligned: &'static str,
    mask_load_unaligned: Box<dyn Fn(String, String) -> String>,

    mask_store_unaligned: &'static str,
    mul_add: &'static str,
    mul: &'static str,
}

struct CplxKernel {
    ty: &'static str,
    reg_ty: &'static str,
    mask_ty: &'static str,
    // register size
    n: usize,
    mr_div_n: usize,
    nr: usize,
    k: Option<usize>,

    target_features: &'static str,
    set1: &'static str,
    swap_re_im: &'static str,

    load_unaligned: &'static str,
    store_unaligned: &'static str,
    mask_load_unaligned: Box<dyn Fn(String, String) -> String>,

    mask_store_unaligned: &'static str,
    mul_addsub: &'static str,
    mul_subadd: &'static str,
    xor: &'static str,
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
                &crate::MicroKernelData {{ alpha, beta, k, dst_cs, lhs_cs, rhs_rs, rhs_cs, last_mask, .. }}: &crate::MicroKernelData< {3:} >,
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

impl Display for CplxKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // function that multiplies (mr_div_n * n, k) by (k, n_r)
        // C += beta * A * B

        // not exactly mr_div_n
        // the actual number of rows is between mr_div_n * (n-1) and mr_div_n * n
        let Self {
            swap_re_im,
            load_unaligned,
            store_unaligned,
            mask_store_unaligned,
            mul_addsub,
            xor,
            ..
        } = self;

        write!(
            f,
            "
            #[target_feature(enable = \"{}\")]\n",
            self.target_features
        )?;
        write!(
            f,
            r#"pub unsafe fn matmul_{0:}_{1:}_{2:}(
                &crate::MicroKernelData {{ alpha, beta, k, dst_cs, lhs_cs, rhs_rs, rhs_cs, last_mask, conj_lhs, conj_rhs }}: &crate::MicroKernelData<num_complex::Complex< {3:} >>,
                dst: *mut num_complex::Complex< {3:} >,
                lhs: *const num_complex::Complex< {3:} >,
                rhs: *const num_complex::Complex< {3:} >,
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

        for idx in 0..2 {
            if idx == 0 {
                write!(f, "if conj_lhs == conj_rhs {{\n")?;
            } else {
                write!(f, "else {{\n")?;
            }

            let mul_add = if idx == 0 {
                self.mul_subadd
            } else {
                self.mul_addsub
            };

            if let Some(k) = self.k {
                for depth in 0..k {
                    write!(f, "let depth = {depth};\n")?;
                    for i in 0..self.mr_div_n {
                        self.write_load_lhs(i, f)?;
                    }

                    for j in 0..self.nr {
                        write!(
                            f,
                            "let tmp_rhs = {}((*rhs.offset(depth * rhs_rs + {j} * rhs_cs)).re);\n",
                            self.set1
                        )?;

                        for i in 0..self.mr_div_n {
                            write!(
                                f,
                                "acc[{i}][{j}] = {mul_add}(tmp_lhs[{i}], tmp_rhs, acc[{i}][{j}]);\n",
                            )?;
                        }
                    }
                    for i in 0..self.mr_div_n {
                        write!(f, "tmp_lhs[{i}] = {}(tmp_lhs[{i}]);", self.swap_re_im)?;
                    }
                    for j in 0..self.nr {
                        write!(
                            f,
                            "let tmp_rhs = {}((*rhs.offset(depth * rhs_rs + {j} * rhs_cs)).im);\n",
                            self.set1
                        )?;

                        for i in 0..self.mr_div_n {
                            write!(
                                f,
                                "acc[{i}][{j}] = {mul_add}(tmp_lhs[{i}], tmp_rhs, acc[{i}][{j}]);\n",
                            )?;
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
                        "let tmp_rhs = {}((*rhs.offset(depth * rhs_rs + {j} * rhs_cs)).re);\n",
                        self.set1
                    )?;

                    for i in 0..self.mr_div_n {
                        write!(
                            f,
                            "acc[{i}][{j}] = {mul_add}(tmp_lhs[{i}], tmp_rhs, acc[{i}][{j}]);\n",
                        )?;
                    }
                }
                for i in 0..self.mr_div_n {
                    write!(f, "tmp_lhs[{i}] = {}(tmp_lhs[{i}]);", self.swap_re_im)?;
                }
                for j in 0..self.nr {
                    write!(
                        f,
                        "let tmp_rhs = {}((*rhs.offset(depth * rhs_rs + {j} * rhs_cs)).im);\n",
                        self.set1
                    )?;

                    for i in 0..self.mr_div_n {
                        write!(
                            f,
                            "acc[{i}][{j}] = {mul_add}(tmp_lhs[{i}], tmp_rhs, acc[{i}][{j}]);\n",
                        )?;
                    }
                }

                write!(f, "}}")?;
            }
            write!(f, "}}")?;
        }

        write!(
            f,
            "let mask = XOR_MASKS[conj_lhs as usize + 2 * conj_rhs as usize];"
        )?;
        for j in 0..self.nr {
            for i in 0..self.mr_div_n {
                write!(f, "acc[{i}][{j}] = core::mem::transmute({xor}(core::mem::transmute(acc[{i}][{j}]), core::mem::transmute(mask)));")?;
            }
        }

        write!(
            f,
            "if alpha == (num_complex::Complex {{ re: 1.0, im: 0.0 }}) {{"
        )?;
        write!(f, "let beta_re = {}(beta.re);\n", self.set1)?;
        write!(f, "let beta_im = {}(beta.im);\n", self.set1)?;
        for j in 0..self.nr {
            for i in 0..self.mr_div_n {
                write!(f, "{{")?;
                write!(
                    f,
                    "let dst = dst.offset({i} * N + {j} * dst_cs) as *mut {};",
                    self.ty,
                )?;
                if i + 1 < self.mr_div_n {
                    write!(
                        f,
                        "{store_unaligned}(
                            dst,
                            {mul_addsub}(
                                {swap_re_im}(acc[{i}][{j}]),
                                beta_im,
                                {mul_addsub}(
                                    acc[{i}][{j}],
                                    beta_re,
                                    {load_unaligned}(dst),
                                ),
                            ),
                        );\n",
                    )?;
                } else {
                    write!(
                        f,
                        "{mask_store_unaligned}(
                            dst,
                            last_mask,
                            {mul_addsub}(
                                {swap_re_im}(acc[{i}][{j}]),
                                beta_im,
                                {mul_addsub}(
                                    acc[{i}][{j}],
                                    beta_re,
                                    {},
                                ),
                            ),
                        );\n",
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

impl CplxKernel {
    fn write_load_lhs(
        &self,
        i: usize,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        Ok(if i + 1 < self.mr_div_n {
            write!(
                f,
                "tmp_lhs[{i}] = {}(lhs.offset(depth * lhs_cs + {i} * N) as *const {});\n",
                self.load_unaligned, self.ty,
            )?;
        } else {
            write!(
                f,
                "tmp_lhs[{i}] = {};\n",
                (self.mask_load_unaligned)(
                    format!("lhs.offset(depth * lhs_cs + {i} * N) as *const {}", self.ty,),
                    "last_mask".to_string()
                ),
            )?;
        })
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
    {
        {
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
        }
        {
            write!(code, "pub mod f64 {{\n")?;
            for mr_div_n in 1..=2 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            ty: "f64",
                            reg_ty: "__m256d",
                            mask_ty: "__m256i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 4,
                            set1: "_mm256_set1_pd",
                            load_unaligned: "_mm256_loadu_pd",
                            store_unaligned: "_mm256_storeu_pd",
                            mask_load_unaligned: Box::new(|ptr, mask| {
                                format!("_mm256_maskload_pd({ptr}, {mask})")
                            }),
                            mask_store_unaligned: "_mm256_maskstore_pd",
                            mul_add: "_mm256_fmadd_pd",
                            mul: "_mm256_mul_pd",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[crate::MicroKernel<f64>; 4]; 2]; 17] = [\n"
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
            pub static MASKS: [crate::__m256i; 4] = unsafe {{ core::mem::transmute([
                [u64::MAX, u64::MAX, u64::MAX, u64::MAX],

                [u64::MAX, 0, 0, 0],
                [u64::MAX, u64::MAX, 0, 0],
                [u64::MAX, u64::MAX, u64::MAX, 0],
            ]) }};
        "
            )?;

            write!(code, "}}\n")?;
        }

        {
            write!(code, "pub mod c32 {{\n")?;
            write!(
                code,
                "const XOR_MASKS: [crate::__m256; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0f32], // no conj
                   [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0f32], // conj lhs
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32], // conj rhs
                   [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0f32], // conj both
                ]) }};\n"
            )?;

            for mr_div_n in 1..=2 {
                for nr in 1..=2 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = CplxKernel {
                            ty: "f32",
                            reg_ty: "__m256",
                            mask_ty: "__m256i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 4,
                            set1: "_mm256_set1_ps",
                            load_unaligned: "_mm256_loadu_ps",
                            store_unaligned: "_mm256_storeu_ps",
                            mask_load_unaligned: Box::new(|ptr, mask| {
                                format!("_mm256_maskload_ps({ptr}, {mask})")
                            }),
                            mask_store_unaligned: "_mm256_maskstore_ps",
                            swap_re_im: "_mm256_permute_ps::<0b10_11_00_01>",
                            mul_addsub: "_mm256_fmsubadd_ps",
                            mul_subadd: "_mm256_fmaddsub_ps",
                            xor: "_mm256_xor_ps",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[crate::MicroKernel<num_complex::Complex<f32>>; 2]; 2]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=2 {
                    write!(code, "[\n")?;
                    for nr in 1..=2 {
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
            pub static MASKS: [crate::__m256i; 4] = unsafe {{ core::mem::transmute([
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],

                [u32::MAX, u32::MAX, 0, 0, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0],
            ]) }};
        "
            )?;

            write!(code, "}}\n")?;
        }

        {
            write!(code, "pub mod c64 {{\n")?;
            write!(
                code,
                "const XOR_MASKS: [crate::__m256d; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0f64], // no conj
                   [ 0.0, -0.0,  0.0, -0.0f64], // conj lhs
                   [ 0.0,  0.0,  0.0,  0.0f64], // conj rhs
                   [-0.0,  0.0, -0.0,  0.0f64], // conj both
                ]) }};\n"
            )?;

            for mr_div_n in 1..=2 {
                for nr in 1..=2 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = CplxKernel {
                            ty: "f64",
                            reg_ty: "__m256d",
                            mask_ty: "__m256i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 2,
                            set1: "_mm256_set1_pd",
                            load_unaligned: "_mm256_loadu_pd",
                            store_unaligned: "_mm256_storeu_pd",
                            mask_load_unaligned: Box::new(|ptr, mask| {
                                format!("_mm256_maskload_pd({ptr}, {mask})")
                            }),
                            mask_store_unaligned: "_mm256_maskstore_pd",
                            swap_re_im: "_mm256_permute_pd::<0b0101>",
                            mul_addsub: "_mm256_fmsubadd_pd",
                            mul_subadd: "_mm256_fmaddsub_pd",
                            xor: "_mm256_xor_pd",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[crate::MicroKernel<num_complex::Complex<f64>>; 2]; 2]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=2 {
                    write!(code, "[\n")?;
                    for nr in 1..=2 {
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
            pub static MASKS: [crate::__m256i; 2] = unsafe {{ core::mem::transmute([
                [u64::MAX, u64::MAX, u64::MAX, u64::MAX],

                [u64::MAX, u64::MAX, 0, 0],
            ]) }};
        "
            )?;

            write!(code, "}}\n")?;
        }
    }
    write!(code, "}}\n")?;
    write!(
        code,
        "
        #[cfg(any(target_arch = \"x86\", target_arch = \"x86_64\"))]
        #[cfg(feature = \"nightly\")]
        pub mod avx512 {{\n"
    )?;
    {
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
    }
    {
        write!(code, "pub mod f64 {{\n")?;
        for mr_div_n in 1..=2 {
            for nr in 1..=4 {
                for k in (1..=16).into_iter().map(Some).chain([None]) {
                    let kernel = RealKernel {
                        ty: "f64",
                        reg_ty: "__m512d",
                        mask_ty: "u8",
                        mr_div_n,
                        nr,
                        k,
                        target_features: "avx512f",
                        n: 8,
                        set1: "_mm512_set1_pd",
                        load_unaligned: "_mm512_loadu_pd",
                        store_unaligned: "_mm512_storeu_pd",
                        mask_load_unaligned: Box::new(|ptr, mask| {
                            format!("_mm512_maskz_loadu_pd({mask}, {ptr})")
                        }),
                        mask_store_unaligned: "_mm512_mask_storeu_pd",
                        mul_add: "_mm512_fmadd_pd",
                        mul: "_mm512_mul_pd",
                    };

                    write!(code, "{kernel}")?;
                }
            }
        }

        write!(
            code,
            "pub static MICROKERNELS: [[[crate::MicroKernel<f64>; 4]; 2]; 17] = [\n"
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
            pub static MASKS: [u8; 8] = [
                0b1111_1111,
                0b0000_0001,
                0b0000_0011,
                0b0000_0111,
                0b0000_1111,
                0b0001_1111,
                0b0011_1111,
                0b0111_1111,
            ];
        "
        )?;

        write!(code, "}}\n")?;
    }
    {
        write!(code, "pub mod c32 {{\n")?;
        write!(
            code,
            "const XOR_MASKS: [crate::__m512; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0f32], // no conj
                   [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0f32], // conj lhs
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32], // conj rhs
                   [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0,-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0f32], // conj both
                ]) }};\n"
        )?;

        for mr_div_n in 1..=2 {
            for nr in 1..=2 {
                for k in (1..=16).into_iter().map(Some).chain([None]) {
                    let kernel = CplxKernel {
                        ty: "f32",
                        reg_ty: "__m512",
                        mask_ty: "u16",
                        mr_div_n,
                        nr,
                        k,
                        target_features: "avx512f",
                        n: 8,
                        set1: "_mm512_set1_ps",
                        load_unaligned: "_mm512_loadu_ps",
                        store_unaligned: "_mm512_storeu_ps",
                        mask_load_unaligned: Box::new(|ptr, mask| {
                            format!("_mm512_maskz_loadu_ps({mask}, {ptr})")
                        }),
                        mask_store_unaligned: "_mm512_mask_storeu_ps",
                        swap_re_im: "_mm512_permute_ps::<0b10_11_00_01>",
                        mul_addsub: "_mm512_fmsubadd_ps",
                        mul_subadd: "_mm512_fmaddsub_ps",
                        xor: "_mm512_xor_si512",
                    };

                    write!(code, "{kernel}")?;
                }
            }
        }

        write!(
                code,
                "pub static MICROKERNELS: [[[crate::MicroKernel<num_complex::Complex<f32>>; 2]; 2]; 17] = [\n"
            )?;
        for k in (1..=16).into_iter().map(Some).chain([None]) {
            write!(code, "[\n")?;
            for mr_div_n in 1..=2 {
                write!(code, "[\n")?;
                for nr in 1..=2 {
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
            pub static MASKS: [u16; 8] = [
                0b1111_1111_1111_1111,
                0b0000_0000_0000_0011,
                0b0000_0000_0000_1111,
                0b0000_0000_0011_1111,
                0b0000_0000_1111_1111,
                0b0000_0011_1111_1111,
                0b0000_1111_1111_1111,
                0b0011_1111_1111_1111,
            ];
        "
        )?;

        write!(code, "}}\n")?;
    }
    {
        write!(code, "pub mod c64 {{\n")?;
        write!(
            code,
            "const XOR_MASKS: [crate::__m512; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0f64], // no conj
                   [ 0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0f64], // conj lhs
                   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0f64], // conj rhs
                   [-0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0f64], // conj both
                ]) }};\n"
        )?;

        for mr_div_n in 1..=2 {
            for nr in 1..=2 {
                for k in (1..=16).into_iter().map(Some).chain([None]) {
                    let kernel = CplxKernel {
                        ty: "f64",
                        reg_ty: "__m512d",
                        mask_ty: "u8",
                        mr_div_n,
                        nr,
                        k,
                        target_features: "avx512f",
                        n: 4,
                        set1: "_mm512_set1_pd",
                        load_unaligned: "_mm512_loadu_pd",
                        store_unaligned: "_mm512_storeu_pd",
                        mask_load_unaligned: Box::new(|ptr, mask| {
                            format!("_mm512_maskz_loadu_pd({mask}, {ptr})")
                        }),
                        mask_store_unaligned: "_mm512_mask_storeu_pd",
                        swap_re_im: "_mm512_permute_pd::<0b01010101>",
                        mul_addsub: "_mm512_fmsubadd_pd",
                        mul_subadd: "_mm512_fmaddsub_pd",
                        xor: "_mm512_xor_si512",
                    };

                    write!(code, "{kernel}")?;
                }
            }
        }

        write!(
                code,
                "pub static MICROKERNELS: [[[crate::MicroKernel<num_complex::Complex<f64>>; 2]; 2]; 17] = [\n"
            )?;
        for k in (1..=16).into_iter().map(Some).chain([None]) {
            write!(code, "[\n")?;
            for mr_div_n in 1..=2 {
                write!(code, "[\n")?;
                for nr in 1..=2 {
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
            pub static MASKS: [u8; 4] = [
                0b1111_1111,
                0b0000_0011,
                0b0000_1111,
                0b0011_1111,
            ];
        "
        )?;

        write!(code, "}}\n")?;
    }
    write!(code, "}}\n")?;

    std::fs::write(&dest_path, format!("{code}")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
