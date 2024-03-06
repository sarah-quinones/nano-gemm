// only targeting scalar, avx2, avx512
// no arm64 (until SVE comes), because no masked load instructions

// f32, f64, c32, c64

use std::fmt::Display;
use std::fmt::Write;

mod generic {
    use super::*;

    pub struct RealKernel {
        pub ty: &'static str,
        pub reg_ty: &'static str,
        // register size
        pub n: usize,
        pub mr: usize,
        pub nr: usize,
        pub k: Option<usize>,

        pub target_features: &'static str,
        pub load_unaligned: [&'static str; 3],
        pub store_unaligned: [&'static str; 3],
        pub set1: &'static str,
        pub mul_add: &'static str,
    }

    pub struct CplxKernel {
        pub ty: &'static str,
        pub reg_ty: &'static str,
        // register size
        pub n: usize,
        pub mr: usize,
        pub nr: usize,
        pub k: Option<usize>,

        pub target_features: &'static str,
        pub load_unaligned: [&'static str; 3],
        pub store_unaligned: [&'static str; 3],
        pub set1: &'static str,
        pub mul_add: &'static str,
        pub conj_mul_add: &'static str,
        pub conj: &'static str,
    }

    impl Display for RealKernel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let Self {
                ty,
                reg_ty,
                n,
                mr,
                nr,
                k,
                target_features,
                load_unaligned,
                store_unaligned,
                mul_add,
                ..
            } = self;

            write!(f, "#[target_feature(enable = \"{target_features}\")]\n")?;
            write!(
                f,
                r#"pub unsafe fn matmul_{mr}_{nr}_{}(
                &nano_gemm_core::MicroKernelData {{ alpha, beta, k, dst_cs, lhs_cs, rhs_rs, rhs_cs, .. }}: &nano_gemm_core::MicroKernelData< {ty} >,
                dst: *mut {ty},
                lhs: *const {ty},
                rhs: *const {ty},
            ) {{
"#,
                k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
            )?;

            write!(f, "_ = k;\n")?;
            let mut i = 0;
            while i < *mr {
                let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);

                for j in 0..*nr {
                    write!(
                        f,
                        "let mut acc_{i}_{j}: {} = core::mem::zeroed();\n",
                        reg_ty
                    )?;
                }

                i += 1 << ii;
            }

            if let Some(k) = self.k {
                for depth in 0..k {
                    write!(f, "let depth = {depth};\n")?;
                    self.inner_kernel(f)?;
                }
            } else {
                write!(f, "for depth in 0..k as isize {{")?;
                self.inner_kernel(f)?;
                write!(f, "}}")?;
            }

            write!(f, "if alpha == 1.0 {{")?;
            write!(f, "let beta = {}(beta);\n", self.set1)?;
            for j in 0..self.nr {
                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(f, "{{")?;
                    write!(f, "let dst = dst.offset({i} + {j} * dst_cs);")?;
                    write!(
                        f,
                        "{}(dst, {mul_add}(beta, acc_{i}_{j}, {}(dst)));\n",
                        store_unaligned[ii], load_unaligned[ii],
                    )?;
                    write!(f, "}}")?;
                    i += 1 << ii;
                }
            }
            write!(f, "}}")?;

            write!(f, "else if alpha == 0.0 {{")?;
            write!(f, "let beta = {}(beta);\n", self.set1)?;
            for j in 0..self.nr {
                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(f, "{{")?;
                    write!(f, "let dst = dst.offset({i} + {j} * dst_cs);")?;
                    write!(
                        f,
                        "{}(dst, {mul_add}(beta, acc_{i}_{j}, core::mem::zeroed()));\n",
                        store_unaligned[ii],
                    )?;
                    write!(f, "}}")?;
                    i += 1 << ii;
                }
            }
            write!(f, "}}")?;

            write!(f, "else {{")?;
            write!(f, "let beta = {}(beta);\n", self.set1)?;
            write!(f, "let alpha = {}(alpha);\n", self.set1)?;
            for j in 0..self.nr {
                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(f, "{{")?;
                    write!(f, "let dst = dst.offset({i} + {j} * dst_cs);")?;
                    write!(
                        f,
                        "{}(dst, {mul_add}(beta, acc_{i}_{j}, {mul_add}(alpha, {}(dst), core::mem::zeroed())));\n",
                        store_unaligned[ii], load_unaligned[ii],
                    )?;
                    write!(f, "}}")?;
                    i += 1 << ii;
                }
            }
            write!(f, "}}")?;

            write!(f, "}}")
        }
    }

    impl RealKernel {
        fn inner_kernel(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            let Self {
                mr,
                set1,
                mul_add,
                load_unaligned,
                n,
                ..
            } = self;

            let mut i = 0;
            while i < *mr {
                let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                write!(
                    f,
                    "let tmp_lhs_{i} = {}(lhs.offset(depth * lhs_cs + {i}));",
                    load_unaligned[ii],
                )?;
                i += 1 << ii;
            }
            for j in 0..self.nr {
                write!(
                    f,
                    "let tmp_rhs = {set1}(*rhs.offset(depth * rhs_rs + {j} * rhs_cs));\n",
                )?;

                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(
                        f,
                        "acc_{i}_{j} = {mul_add}(tmp_lhs_{i}, tmp_rhs, acc_{i}_{j});\n",
                    )?;
                    i += 1 << ii;
                }
            }

            Ok(())
        }
    }

    impl Display for CplxKernel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let Self {
                ty,
                reg_ty,
                n,
                mr,
                nr,
                k,
                target_features,
                load_unaligned,
                store_unaligned,
                mul_add,
                conj,
                ..
            } = self;

            write!(f, "#[target_feature(enable = \"{target_features}\")]\n")?;
            write!(
                f,
                r#"pub unsafe fn matmul_{mr}_{nr}_{}(
                &nano_gemm_core::MicroKernelData {{ alpha, beta, k, dst_cs, lhs_cs, rhs_rs, rhs_cs, conj_lhs, conj_rhs, .. }}: &nano_gemm_core::MicroKernelData< {ty} >,
                dst: *mut {ty},
                lhs: *const {ty},
                rhs: *const {ty},
            ) {{
"#,
                k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
            )?;

            write!(f, "_ = k;\n")?;
            let mut i = 0;
            while i < *mr {
                let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);

                for j in 0..*nr {
                    write!(
                        f,
                        "let mut acc_{i}_{j}: {} = core::mem::zeroed();\n",
                        reg_ty
                    )?;
                }

                i += 1 << ii;
            }

            write!(f, "if conj_lhs == conj_rhs {{")?;
            if let Some(k) = self.k {
                for depth in 0..k {
                    write!(f, "let depth = {depth};\n")?;
                    self.inner_kernel_no_conj(f)?;
                }
            } else {
                write!(f, "for depth in 0..k as isize {{")?;
                self.inner_kernel_no_conj(f)?;
                write!(f, "}}")?;
            }
            write!(f, "}} else {{")?;
            if let Some(k) = self.k {
                for depth in 0..k {
                    write!(f, "let depth = {depth};\n")?;
                    self.inner_kernel_conj(f)?;
                }
            } else {
                write!(f, "for depth in 0..k as isize {{")?;
                self.inner_kernel_conj(f)?;
                write!(f, "}}")?;
            }
            write!(f, "}}")?;

            write!(f, "if conj_rhs {{")?;
            for j in 0..self.nr {
                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(f, "acc_{i}_{j} = {conj}(acc_{i}_{j});")?;
                    i += 1 << ii;
                }
            }
            write!(f, "}}")?;

            write!(f, "if alpha == ({ty} {{ re: 1.0, im: 0.0 }}) {{")?;
            write!(f, "let beta = {}(beta);\n", self.set1)?;
            for j in 0..self.nr {
                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(f, "{{")?;
                    write!(f, "let dst = dst.offset({i} + {j} * dst_cs);")?;
                    write!(
                        f,
                        "{}(dst, {mul_add}(beta, acc_{i}_{j}, {}(dst)));\n",
                        store_unaligned[ii], load_unaligned[ii],
                    )?;
                    write!(f, "}}")?;
                    i += 1 << ii;
                }
            }
            write!(f, "}}")?;

            write!(f, "else if alpha == ({ty} {{ re: 0.0, im: 0.0 }}) {{")?;
            write!(f, "let beta = {}(beta);\n", self.set1)?;
            for j in 0..self.nr {
                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(f, "{{")?;
                    write!(f, "let dst = dst.offset({i} + {j} * dst_cs);")?;
                    write!(
                        f,
                        "{}(dst, {mul_add}(beta, acc_{i}_{j}, core::mem::zeroed()));\n",
                        store_unaligned[ii],
                    )?;
                    write!(f, "}}")?;
                    i += 1 << ii;
                }
            }
            write!(f, "}}")?;

            write!(f, "else {{")?;
            write!(f, "let beta = {}(beta);\n", self.set1)?;
            write!(f, "let alpha = {}(alpha);\n", self.set1)?;
            for j in 0..self.nr {
                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(f, "{{")?;
                    write!(f, "let dst = dst.offset({i} + {j} * dst_cs);")?;
                    write!(
                        f,
                        "{}(dst, {mul_add}(beta, acc_{i}_{j}, {mul_add}(alpha, {}(dst), core::mem::zeroed())));\n",
                        store_unaligned[ii], load_unaligned[ii],
                    )?;
                    write!(f, "}}")?;
                    i += 1 << ii;
                }
            }
            write!(f, "}}")?;

            write!(f, "}}")
        }
    }

    impl CplxKernel {
        fn inner_kernel_no_conj(
            &self,
            f: &mut std::fmt::Formatter<'_>,
        ) -> Result<(), std::fmt::Error> {
            let Self {
                mr,
                set1,
                mul_add,
                load_unaligned,
                n,
                ..
            } = self;

            let mut i = 0;
            while i < *mr {
                let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                write!(
                    f,
                    "let tmp_lhs_{i} = {}(lhs.offset(depth * lhs_cs + {i}));",
                    load_unaligned[ii],
                )?;
                i += 1 << ii;
            }
            for j in 0..self.nr {
                write!(
                    f,
                    "let tmp_rhs = {set1}(*rhs.offset(depth * rhs_rs + {j} * rhs_cs));\n",
                )?;

                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(
                        f,
                        "acc_{i}_{j} = {mul_add}(tmp_lhs_{i}, tmp_rhs, acc_{i}_{j});\n",
                    )?;
                    i += 1 << ii;
                }
            }

            Ok(())
        }

        fn inner_kernel_conj(
            &self,
            f: &mut std::fmt::Formatter<'_>,
        ) -> Result<(), std::fmt::Error> {
            let Self {
                mr,
                set1,
                conj_mul_add,
                load_unaligned,
                n,
                ..
            } = self;

            let mut i = 0;
            while i < *mr {
                let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                write!(
                    f,
                    "let tmp_lhs_{i} = {}(lhs.offset(depth * lhs_cs + {i}));",
                    load_unaligned[ii],
                )?;
                i += 1 << ii;
            }
            for j in 0..self.nr {
                write!(
                    f,
                    "let tmp_rhs = {set1}(*rhs.offset(depth * rhs_rs + {j} * rhs_cs));\n",
                )?;

                let mut i = 0;
                while i < *mr {
                    let ii = Ord::min((mr - i).ilog2() as usize, n.ilog2() as usize);
                    write!(
                        f,
                        "acc_{i}_{j} = {conj_mul_add}(tmp_lhs_{i}, tmp_rhs, acc_{i}_{j});\n",
                    )?;
                    i += 1 << ii;
                }
            }

            Ok(())
        }
    }
}

pub mod aarch64 {
    use super::*;
    use generic::{CplxKernel, RealKernel};

    pub fn codegen_f32() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod f32 {{\n")?;
        write!(code, "pub mod neon {{\n")?;
        write!(
            code,
            r###"
            use core::arch::aarch64::*;
            use core::mem::transmute;
            use core::mem::transmute_copy;

            #[inline(always)]
            unsafe fn set1(v: f32) -> float32x4_t {{
                transmute([v; 4])
            }}
            #[inline(always)]
            unsafe fn mul_add(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {{
                vmlaq_f32(c, a, b)
            }}
            #[inline(always)]
            unsafe fn load_1(ptr: *const f32) -> float32x4_t {{
                transmute([*ptr; 4])
            }}
            #[inline(always)]
            unsafe fn load_2(ptr: *const f32) -> float32x4_t {{
                transmute([*(ptr as *const [f32; 2]); 2])
            }}
            #[inline(always)]
            unsafe fn load_4(ptr: *const f32) -> float32x4_t {{
                transmute(*(ptr as *const [f32; 4]))
            }}

            #[inline(always)]
            unsafe fn store_1(ptr: *mut f32, v: float32x4_t) {{
                *(ptr as *mut [f32; 1]) = transmute_copy(&v);
            }}
            #[inline(always)]
            unsafe fn store_2(ptr: *mut f32, v: float32x4_t) {{
                *(ptr as *mut [f32; 2]) = transmute_copy(&v);
            }}
            #[inline(always)]
            unsafe fn store_4(ptr: *mut f32, v: float32x4_t) {{
                *(ptr as *mut [f32; 4]) = transmute_copy(&v);
            }}
            "###
        )?;
        for mr in 1..=8 {
            for nr in 1..=4 {
                for k in (1..=16).into_iter().map(Some).chain([None]) {
                    let kernel = RealKernel {
                        ty: "f32",
                        reg_ty: "float32x4_t",
                        n: 4,
                        mr,
                        nr,
                        k,
                        target_features: "neon",
                        load_unaligned: ["load_1", "load_2", "load_4"],
                        store_unaligned: ["store_1", "store_2", "store_4"],
                        set1: "set1",
                        mul_add: "mul_add",
                    };
                    write!(code, "{kernel}")?;
                }
            }
        }

        write!(
            code,
            "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f32>; 4]; 8]; 17] = [\n"
        )?;
        for k in (1..=16).into_iter().map(Some).chain([None]) {
            write!(code, "[\n")?;
            for mr in 1..=8 {
                write!(code, "[\n")?;
                for nr in 1..=4 {
                    write!(
                        code,
                        "matmul_{mr}_{nr}_{},",
                        k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
                    )?;
                }
                write!(code, "],\n")?;
            }
            write!(code, "],\n")?;
        }
        write!(code, "];\n")?;
        write!(code, "}}")?;
        write!(code, "}}")?;

        Ok(code)
    }

    pub fn codegen_f64() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod f64 {{\n")?;
        write!(code, "pub mod neon {{\n")?;
        write!(
            code,
            r###"
            use core::arch::aarch64::*;
            use core::mem::transmute;
            use core::mem::transmute_copy;

            #[inline(always)]
            unsafe fn set1(v: f64) -> float64x2_t {{
                transmute([v; 2])
            }}
            #[inline(always)]
            unsafe fn mul_add(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {{
                vmlaq_f64(c, a, b)
            }}
            #[inline(always)]
            unsafe fn load_1(ptr: *const f64) -> float64x2_t {{
                transmute([*ptr; 2])
            }}
            #[inline(always)]
            unsafe fn load_2(ptr: *const f64) -> float64x2_t {{
                transmute(*(ptr as *const [f64; 2]))
            }}

            #[inline(always)]
            unsafe fn store_1(ptr: *mut f64, v: float64x2_t) {{
                *(ptr as *mut [f64; 1]) = transmute_copy(&v);
            }}
            #[inline(always)]
            unsafe fn store_2(ptr: *mut f64, v: float64x2_t) {{
                *(ptr as *mut [f64; 2]) = transmute_copy(&v);
            }}
            "###
        )?;
        for mr in 1..=4 {
            for nr in 1..=4 {
                for k in (1..=16).into_iter().map(Some).chain([None]) {
                    let kernel = RealKernel {
                        ty: "f64",
                        reg_ty: "float64x2_t",
                        n: 2,
                        mr,
                        nr,
                        k,
                        target_features: "neon",
                        load_unaligned: ["load_1", "load_2", "load_4"],
                        store_unaligned: ["store_1", "store_2", "store_4"],
                        set1: "set1",
                        mul_add: "mul_add",
                    };
                    write!(code, "{kernel}")?;
                }
            }
        }

        write!(
            code,
            "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f64>; 4]; 4]; 17] = [\n"
        )?;
        for k in (1..=16).into_iter().map(Some).chain([None]) {
            write!(code, "[\n")?;
            for mr in 1..=4 {
                write!(code, "[\n")?;
                for nr in 1..=4 {
                    write!(
                        code,
                        "matmul_{mr}_{nr}_{},",
                        k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
                    )?;
                }
                write!(code, "],\n")?;
            }
            write!(code, "],\n")?;
        }
        write!(code, "];\n")?;
        write!(code, "}}")?;
        write!(code, "}}")?;

        Ok(code)
    }

    pub fn codegen_c32() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod c32 {{\n")?;
        write!(code, "pub mod neon {{ use crate::c32;\n")?;
        write!(
            code,
            r###"
            use core::arch::aarch64::*;
            use core::mem::transmute;
            use core::mem::transmute_copy;

            #[inline(always)]
            unsafe fn set1(v: c32) -> float32x4_t {{
                transmute([v; 2])
            }}
            #[inline(always)]
            unsafe fn mul_add(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {{
                vcmlaq_90_f32(vcmlaq_0_f32(c, a, b), a, b)
            }}
            #[inline(always)]
            unsafe fn conj_mul_add(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {{
                vcmlaq_270_f32(vcmlaq_0_f32(c, a, b), a, b)
            }}
            #[inline(always)]
            unsafe fn conj(a: float32x4_t) -> float32x4_t {{
                transmute(veorq_u32(transmute(a), transmute([0.0, -0.0, 0.0, -0.0f32])))
            }}
            #[inline(always)]
            unsafe fn load_1(ptr: *const c32) -> float32x4_t {{
                transmute([*ptr; 2])
            }}
            #[inline(always)]
            unsafe fn load_2(ptr: *const c32) -> float32x4_t {{
                transmute(*(ptr as *const [c32; 2]))
            }}

            #[inline(always)]
            unsafe fn store_1(ptr: *mut c32, v: float32x4_t) {{
                *(ptr as *mut [c32; 1]) = transmute_copy(&v);
            }}
            #[inline(always)]
            unsafe fn store_2(ptr: *mut c32, v: float32x4_t) {{
                *(ptr as *mut [c32; 2]) = transmute_copy(&v);
            }}

#[inline]
#[target_feature(enable = "neon,fcma")]
unsafe fn vcmlaq_0_f32(mut acc: float32x4_t, lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {{
    core::arch::asm!(
        "fcmla {{0:v}}.4s, {{1:v}}.4s, {{2:v}}.4s, 0",
        inout(vreg) acc,
        in(vreg) lhs,
        in(vreg) rhs,
        options(pure, nomem, nostack));
    acc
}}

#[inline]
#[target_feature(enable = "neon,fcma")]
unsafe fn vcmlaq_90_f32(mut acc: float32x4_t, lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {{
    core::arch::asm!(
        "fcmla {{0:v}}.4s, {{1:v}}.4s, {{2:v}}.4s, 90",
        inout(vreg) acc,
        in(vreg) lhs,
        in(vreg) rhs,
        options(pure, nomem, nostack));
    acc
}}

#[inline]
#[target_feature(enable = "neon,fcma")]
unsafe fn vcmlaq_270_f32(mut acc: float32x4_t, lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {{
    core::arch::asm!(
        "fcmla {{0:v}}.4s, {{1:v}}.4s, {{2:v}}.4s, 270",
        inout(vreg) acc,
        in(vreg) lhs,
        in(vreg) rhs,
        options(pure, nomem, nostack));
    acc
}}
            "###
        )?;
        for mr in 1..=4 {
            for nr in 1..=4 {
                for k in (1..=16).into_iter().map(Some).chain([None]) {
                    let kernel = CplxKernel {
                        ty: "c32",
                        reg_ty: "float32x4_t",
                        n: 2,
                        mr,
                        nr,
                        k,
                        target_features: "neon,fcma",
                        load_unaligned: ["load_1", "load_2", "load_4"],
                        store_unaligned: ["store_1", "store_2", "store_4"],
                        set1: "set1",
                        mul_add: "mul_add",
                        conj_mul_add: "conj_mul_add",
                        conj: "conj",
                    };
                    write!(code, "{kernel}")?;
                }
            }
        }

        write!(
            code,
            "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<c32>; 4]; 4]; 17] = [\n"
        )?;
        for k in (1..=16).into_iter().map(Some).chain([None]) {
            write!(code, "[\n")?;
            for mr in 1..=4 {
                write!(code, "[\n")?;
                for nr in 1..=4 {
                    write!(
                        code,
                        "matmul_{mr}_{nr}_{},",
                        k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
                    )?;
                }
                write!(code, "],\n")?;
            }
            write!(code, "],\n")?;
        }
        write!(code, "];\n")?;
        write!(code, "}}")?;
        write!(code, "}}")?;

        Ok(code)
    }

    pub fn codegen_c64() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod c64 {{\n")?;
        write!(code, "pub mod neon {{ use crate::c64;\n")?;
        write!(
            code,
            r###"
            use core::arch::aarch64::*;
            use core::mem::transmute;

            #[inline(always)]
            unsafe fn set1(v: c64) -> float64x2_t {{
                transmute(v)
            }}
            #[inline(always)]
            unsafe fn mul_add(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {{
                vcmlaq_90_f64(vcmlaq_0_f64(c, a, b), a, b)
            }}
            #[inline(always)]
            unsafe fn conj_mul_add(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {{
                vcmlaq_270_f64(vcmlaq_0_f64(c, a, b), a, b)
            }}
            #[inline(always)]
            unsafe fn conj(a: float64x2_t) -> float64x2_t {{
                transmute(veorq_u64(transmute(a), transmute([0.0, -0.0f64])))
            }}
            #[inline(always)]
            unsafe fn load_1(ptr: *const c64) -> float64x2_t {{
                transmute(*ptr)
            }}

            #[inline(always)]
            unsafe fn store_1(ptr: *mut c64, v: float64x2_t) {{
                *ptr = transmute(v);
            }}

#[inline]
#[target_feature(enable = "neon,fcma")]
unsafe fn vcmlaq_0_f64(mut acc: float64x2_t, lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {{
    core::arch::asm!(
        "fcmla {{0:v}}.2d, {{1:v}}.2d, {{2:v}}.2d, 0",
        inout(vreg) acc,
        in(vreg) lhs,
        in(vreg) rhs,
        options(pure, nomem, nostack));
    acc
}}

#[inline]
#[target_feature(enable = "neon,fcma")]
unsafe fn vcmlaq_90_f64(mut acc: float64x2_t, lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {{
    core::arch::asm!(
        "fcmla {{0:v}}.2d, {{1:v}}.2d, {{2:v}}.2d, 90",
        inout(vreg) acc,
        in(vreg) lhs,
        in(vreg) rhs,
        options(pure, nomem, nostack));
    acc
}}

#[inline]
#[target_feature(enable = "neon,fcma")]
unsafe fn vcmlaq_270_f64(mut acc: float64x2_t, lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {{
    core::arch::asm!(
        "fcmla {{0:v}}.2d, {{1:v}}.2d, {{2:v}}.2d, 270",
        inout(vreg) acc,
        in(vreg) lhs,
        in(vreg) rhs,
        options(pure, nomem, nostack));
    acc
}}
            "###
        )?;
        for mr in 1..=2 {
            for nr in 1..=4 {
                for k in (1..=16).into_iter().map(Some).chain([None]) {
                    let kernel = CplxKernel {
                        ty: "c64",
                        reg_ty: "float64x2_t",
                        n: 1,
                        mr,
                        nr,
                        k,
                        target_features: "neon,fcma",
                        load_unaligned: ["load_1", "load_2", "load_4"],
                        store_unaligned: ["store_1", "store_2", "store_4"],
                        set1: "set1",
                        mul_add: "mul_add",
                        conj_mul_add: "conj_mul_add",
                        conj: "conj",
                    };
                    write!(code, "{kernel}")?;
                }
            }
        }

        write!(
            code,
            "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<c64>; 4]; 2]; 17] = [\n"
        )?;
        for k in (1..=16).into_iter().map(Some).chain([None]) {
            write!(code, "[\n")?;
            for mr in 1..=2 {
                write!(code, "[\n")?;
                for nr in 1..=4 {
                    write!(
                        code,
                        "matmul_{mr}_{nr}_{},",
                        k.map(|k| k.to_string()).unwrap_or("dyn".to_string()),
                    )?;
                }
                write!(code, "],\n")?;
            }
            write!(code, "],\n")?;
        }
        write!(code, "];\n")?;
        write!(code, "}}")?;
        write!(code, "}}")?;

        Ok(code)
    }
}

pub mod x86 {
    use super::*;

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
        need_mask: bool,
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
        need_mask: bool,
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
                &nano_gemm_core::MicroKernelData {{ alpha, beta, k, dst_cs, lhs_cs, rhs_rs, rhs_cs, last_mask, .. }}: &nano_gemm_core::MicroKernelData< {3:} >,
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

            if self.need_mask {
                write!(
                    f,
                    "let last_mask = *(last_mask as *const {});\n",
                    self.mask_ty
                )?;
            } else {
                write!(f, "_ = last_mask;\n")?;
            }

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
                                write!(
                                    f,
                                    "acc[{i}][{j}] = {}(tmp_lhs[{i}], tmp_rhs);\n",
                                    self.mul
                                )?;
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
                    if i + 1 < self.mr_div_n || !self.need_mask {
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
                    if i + 1 < self.mr_div_n || !self.need_mask {
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
                    if i + 1 < self.mr_div_n || !self.need_mask {
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
                &nano_gemm_core::MicroKernelData {{ alpha, beta, k, dst_cs, lhs_cs, rhs_rs, rhs_cs, last_mask, conj_lhs, conj_rhs }}: &nano_gemm_core::MicroKernelData<num_complex::Complex< {3:} >>,
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

            if self.need_mask {
                write!(
                    f,
                    "let last_mask = *(last_mask as *const {});\n",
                    self.mask_ty
                )?;
            } else {
                write!(f, "_ = last_mask;\n")?;
            }

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
                    if i + 1 < self.mr_div_n || !self.need_mask {
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

            write!(
                f,
                "else if alpha == (num_complex::Complex {{ re: 0.0, im: 0.0 }}) {{"
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
                    if i + 1 < self.mr_div_n || !self.need_mask {
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
                                    core::mem::zeroed(),
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
                                    core::mem::zeroed(),
                                ),
                            ),
                        );\n"
                        )?;
                    }
                    write!(f, "}}")?;
                }
            }
            write!(f, "}}")?;
            write!(f, "else {{")?;
            write!(f, "let beta_re = {}(beta.re);\n", self.set1)?;
            write!(f, "let beta_im = {}(beta.im);\n", self.set1)?;
            write!(f, "let alpha_re = {}(alpha.re);\n", self.set1)?;
            write!(f, "let alpha_im = {}(alpha.im);\n", self.set1)?;
            for j in 0..self.nr {
                for i in 0..self.mr_div_n {
                    write!(f, "{{")?;
                    write!(
                        f,
                        "let dst = dst.offset({i} * N + {j} * dst_cs) as *mut {};",
                        self.ty,
                    )?;
                    if i + 1 < self.mr_div_n || !self.need_mask {
                        write!(
                            f,
                            "let dst_conj = core::mem::transmute({xor}(
                            core::mem::transmute({load_unaligned}(dst)),
                            core::mem::transmute(XOR_MASKS[1]),
                        ));"
                        )?;

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
                                    {mul_addsub}(
                                        {swap_re_im}(dst_conj),
                                        alpha_im,
                                        {mul_addsub}(
                                            dst_conj,
                                            alpha_re,
                                            core::mem::zeroed(),
                                        ),
                                    ),
                                ),
                            ),
                        );\n",
                        )?;
                    } else {
                        write!(
                            f,
                            "let dst_conj = core::mem::transmute({xor}(
                            core::mem::transmute({}),
                            core::mem::transmute(XOR_MASKS[1]),
                        ));",
                            (self.mask_load_unaligned)(format!("dst"), "last_mask".to_string())
                        )?;

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
                                    {mul_addsub}(
                                        {swap_re_im}(dst_conj),
                                        alpha_im,
                                        {mul_addsub}(
                                            dst_conj,
                                            alpha_re,
                                            core::mem::zeroed(),
                                        ),
                                    ),
                                ),
                            ),
                        );\n"
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
            Ok(if i + 1 < self.mr_div_n || !self.need_mask {
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
            Ok(if i + 1 < self.mr_div_n || !self.need_mask {
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

    pub fn codegen_f32() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod f32 {{\n")?;
        write!(code, "pub mod f32x1 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 4; pub const N: usize = 1;"
            )?;

            for mr_div_n in 1..=1 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            ty: "f32",
                            reg_ty: "__m128",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 1,
                            set1: "crate::x86::splat_1s",
                            load_unaligned: "_mm_load_ss",
                            store_unaligned: "_mm_store_ss",
                            mask_load_unaligned: Box::new(|_, _| String::new()),
                            mask_store_unaligned: "",
                            mul_add: "_mm_fmadd_ss",
                            mul: "_mm_mul_ss",
                            need_mask: false,
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f32>; 4]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
        }
        write!(code, "}}\n")?;
        write!(code, "pub mod f32x2 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 4; pub const N: usize = 2;"
            )?;
            for mr_div_n in 1..=1 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            need_mask: false,
                            ty: "f32",
                            reg_ty: "__m128",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 2,
                            set1: "_mm_set1_ps",
                            load_unaligned: "crate::x86::load_2s",
                            store_unaligned: "crate::x86::store_2s",
                            mask_load_unaligned: Box::new(|_, _| String::new()),
                            mask_store_unaligned: "",
                            mul_add: "_mm_fmadd_ps",
                            mul: "_mm_mul_ps",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f32>; 4]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
        }
        write!(code, "}}\n")?;

        write!(code, "pub mod f32x4 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 4; pub const N: usize = 4;"
            )?;
            for mr_div_n in 1..=1 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            need_mask: true,
                            ty: "f32",
                            reg_ty: "__m128",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 4,
                            set1: "_mm_set1_ps",
                            load_unaligned: "_mm_loadu_ps",
                            store_unaligned: "_mm_storeu_ps",
                            mask_load_unaligned: Box::new(|ptr, mask| {
                                format!("_mm_maskload_ps({ptr}, {mask})")
                            }),
                            mask_store_unaligned: "_mm_maskstore_ps",
                            mul_add: "_mm_fmadd_ps",
                            mul: "_mm_mul_ps",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f32>; 4]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
            pub static MASKS: [crate::x86::__m128i; 4] = unsafe {{ core::mem::transmute([
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX],

                [u32::MAX, 0, 0, 0],
                [u32::MAX, u32::MAX, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, 0],
            ]) }};
        "
            )?;
        }
        write!(code, "}}\n")?;
        write!(code, "pub mod avx {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 4; pub const N: usize = 8;"
            )?;

            for mr_div_n in 1..=2 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            need_mask: true,
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
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f32>; 4]; 2]; 17] = [\n"
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
            pub static MASKS: [crate::x86::__m256i; 8] = unsafe {{ core::mem::transmute([
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
        }
        write!(code, "}}\n")?;

        write!(code, "#[cfg(feature = \"nightly\")] pub mod avx512 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 4; pub const N: usize = 16;"
            )?;

            for mr_div_n in 1..=2 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            need_mask: true,
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
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f32>; 4]; 2]; 17] = [\n"
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
        }
        write!(code, "}}\n")?;
        write!(code, "}}\n")?;

        Ok(code)
    }

    pub fn codegen_f64() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod f64 {{\n")?;
        write!(code, "pub mod f64x1 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 4; pub const N: usize = 1;"
            )?;

            for mr_div_n in 1..=1 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            need_mask: false,
                            ty: "f64",
                            reg_ty: "__m128d",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 1,
                            set1: "crate::x86::splat_1d",
                            load_unaligned: "_mm_load_sd",
                            store_unaligned: "_mm_store_sd",
                            mask_load_unaligned: Box::new(|_, _| String::new()),
                            mask_store_unaligned: "",
                            mul_add: "_mm_fmadd_sd",
                            mul: "_mm_mul_sd",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f64>; 4]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
        }
        write!(code, "}}\n")?;
        write!(code, "pub mod f64x2 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 4; pub const N: usize = 2;"
            )?;
            for mr_div_n in 1..=1 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            need_mask: false,
                            ty: "f64",
                            reg_ty: "__m128d",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 2,
                            set1: "_mm_set1_pd",
                            load_unaligned: "_mm_loadu_pd",
                            store_unaligned: "_mm_storeu_pd",
                            mask_load_unaligned: Box::new(|_, _| String::new()),
                            mask_store_unaligned: "",
                            mul_add: "_mm_fmadd_pd",
                            mul: "_mm_mul_pd",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f64>; 4]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
        }
        write!(code, "}}\n")?;

        write!(
            code,
            "
        pub mod avx {{\n"
        )?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 4; pub const N: usize = 4;"
            )?;

            {
                for mr_div_n in 1..=2 {
                    for nr in 1..=4 {
                        for k in (1..=16).into_iter().map(Some).chain([None]) {
                            let kernel = RealKernel {
                                need_mask: true,
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
                    "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f64>; 4]; 2]; 17] = [\n"
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
            pub static MASKS: [crate::x86::__m256i; 4] = unsafe {{ core::mem::transmute([
                [u64::MAX, u64::MAX, u64::MAX, u64::MAX],

                [u64::MAX, 0, 0, 0],
                [u64::MAX, u64::MAX, 0, 0],
                [u64::MAX, u64::MAX, u64::MAX, 0],
            ]) }};
        "
                )?;
            }
        }
        write!(code, "}}\n")?;

        write!(
            code,
            "
        #[cfg(feature = \"nightly\")]
        pub mod avx512 {{\n"
        )?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 4; pub const N: usize = 8;"
            )?;

            for mr_div_n in 1..=2 {
                for nr in 1..=4 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = RealKernel {
                            need_mask: true,
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
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<f64>; 4]; 2]; 17] = [\n"
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
        }
        write!(code, "}}\n")?;
        write!(code, "}}\n")?;

        Ok(code)
    }

    pub fn codegen_c32() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod c32 {{\n")?;
        write!(code, "pub mod c32x1 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 2; pub const N: usize = 1;"
            )?;
            write!(
                code,
                "const XOR_MASKS: [crate::x86::__m128; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0f32], // no conj
                   [ 0.0, -0.0,  0.0, -0.0f32], // conj lhs
                   [ 0.0,  0.0,  0.0,  0.0f32], // conj rhs
                   [-0.0,  0.0, -0.0,  0.0f32], // conj both
                ]) }};\n"
            )?;

            for mr_div_n in 1..=1 {
                for nr in 1..=2 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = CplxKernel {
                            need_mask: false,
                            ty: "f32",
                            reg_ty: "__m128",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 1,
                            set1: "_mm_set1_ps",
                            load_unaligned: "crate::x86::load_2s",
                            store_unaligned: "crate::x86::store_2s",
                            mask_load_unaligned: Box::new(|ptr, mask| {
                                format!("_mm_maskload_ps({ptr}, {mask})")
                            }),
                            mask_store_unaligned: "_mm_maskstore_ps",
                            swap_re_im: "_mm_permute_ps::<0b10_11_00_01>",
                            mul_addsub: "_mm_fmsubadd_ps",
                            mul_subadd: "_mm_fmaddsub_ps",
                            xor: "_mm_xor_ps",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<num_complex::Complex<f32>>; 2]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
        }
        write!(code, "}}\n")?;

        write!(code, "pub mod c32x2 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 2; pub const N: usize = 2;"
            )?;
            write!(
                code,
                "const XOR_MASKS: [crate::x86::__m128; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0f32], // no conj
                   [ 0.0, -0.0,  0.0, -0.0f32], // conj lhs
                   [ 0.0,  0.0,  0.0,  0.0f32], // conj rhs
                   [-0.0,  0.0, -0.0,  0.0f32], // conj both
                ]) }};\n"
            )?;

            for mr_div_n in 1..=1 {
                for nr in 1..=2 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = CplxKernel {
                            need_mask: false,
                            ty: "f32",
                            reg_ty: "__m128",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 2,
                            set1: "_mm_set1_ps",
                            load_unaligned: "_mm_loadu_ps",
                            store_unaligned: "_mm_storeu_ps",
                            mask_load_unaligned: Box::new(|ptr, mask| {
                                format!("_mm_maskload_ps({ptr}, {mask})")
                            }),
                            mask_store_unaligned: "_mm_maskstore_ps",
                            swap_re_im: "_mm_permute_ps::<0b10_11_00_01>",
                            mul_addsub: "_mm_fmsubadd_ps",
                            mul_subadd: "_mm_fmaddsub_ps",
                            xor: "_mm_xor_ps",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<num_complex::Complex<f32>>; 2]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
        }
        write!(code, "}}\n")?;
        write!(code, " pub mod avx {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 2; pub const N: usize = 4;"
            )?;
            write!(
                code,
                "const XOR_MASKS: [crate::x86::__m256; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0f32], // no conj
                   [ 0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0f32], // conj lhs
                   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0f32], // conj rhs
                   [-0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0f32], // conj both
                ]) }};\n"
            )?;

            for mr_div_n in 1..=2 {
                for nr in 1..=2 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = CplxKernel {
                            need_mask: true,
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
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<num_complex::Complex<f32>>; 2]; 2]; 17] = [\n"
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
            pub static MASKS: [crate::x86::__m256i; 4] = unsafe {{ core::mem::transmute([
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],

                [u32::MAX, u32::MAX, 0, 0, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0, 0, 0],
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0, 0],
            ]) }};
        "
            )?;
        }
        write!(code, "}}\n")?;
        write!(
            code,
            "
        #[cfg(feature = \"nightly\")]
        pub mod avx512 {{\n"
        )?;

        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 2; pub const N: usize = 8;"
            )?;
            write!(
            code,
            "const XOR_MASKS: [crate::x86::__m512; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0f32], // no conj
                   [ 0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0f32], // conj lhs
                   [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0f32], // conj rhs
                   [-0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0, -0.0,  0.0f32], // conj both
                ]) }};\n"
            )?;

            for mr_div_n in 1..=2 {
                for nr in 1..=2 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = CplxKernel {
                            need_mask: true,
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
                            mul_addsub: "crate::x86::subadd_ps",
                            mul_subadd: "_mm512_fmaddsub_ps",
                            xor: "_mm512_xor_si512",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<num_complex::Complex<f32>>; 2]; 2]; 17] = [\n"
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
        }
        write!(code, "}}\n")?;
        write!(code, "}}\n")?;

        Ok(code)
    }

    pub fn codegen_c64() -> Result<String, Box<dyn std::error::Error>> {
        let mut code = String::new();

        write!(code, "pub mod c64 {{\n")?;
        write!(code, "pub mod c64x1 {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 1; pub const NR: usize = 2; pub const N: usize = 1;"
            )?;
            write!(
                code,
                "const XOR_MASKS: [crate::x86::__m128d; 4] = unsafe {{ core::mem::transmute([
                   [-0.0, -0.0f64], // no conj
                   [ 0.0, -0.0f64], // conj lhs
                   [ 0.0,  0.0f64], // conj rhs
                   [-0.0,  0.0f64], // conj both
                ]) }};\n"
            )?;

            for mr_div_n in 1..=1 {
                for nr in 1..=2 {
                    for k in (1..=16).into_iter().map(Some).chain([None]) {
                        let kernel = CplxKernel {
                            need_mask: false,
                            ty: "f64",
                            reg_ty: "__m128d",
                            mask_ty: "__m128i",
                            mr_div_n,
                            nr,
                            k,
                            target_features: "avx,avx2,fma",
                            n: 1,
                            set1: "_mm_set1_pd",
                            load_unaligned: "_mm_loadu_pd",
                            store_unaligned: "_mm_storeu_pd",
                            mask_load_unaligned: Box::new(|ptr, mask| {
                                format!("_mm_maskload_pd({ptr}, {mask})")
                            }),
                            mask_store_unaligned: "_mm_maskstore_pd",
                            swap_re_im: "_mm_permute_pd::<0b01>",
                            mul_addsub: "_mm_fmsubadd_pd",
                            mul_subadd: "_mm_fmaddsub_pd",
                            xor: "_mm_xor_pd",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<num_complex::Complex<f64>>; 2]; 1]; 17] = [\n"
            )?;
            for k in (1..=16).into_iter().map(Some).chain([None]) {
                write!(code, "[\n")?;
                for mr_div_n in 1..=1 {
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
        }
        write!(code, "}}\n")?;

        write!(code, "pub mod avx {{\n")?;
        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 2; pub const N: usize = 2;"
            )?;
            write!(
                code,
                "const XOR_MASKS: [crate::x86::__m256d; 4] = unsafe {{ core::mem::transmute([
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
                            need_mask: true,
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
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<num_complex::Complex<f64>>; 2]; 2]; 17] = [\n"
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
            pub static MASKS: [crate::x86::__m256i; 2] = unsafe {{ core::mem::transmute([
                [u64::MAX, u64::MAX, u64::MAX, u64::MAX],

                [u64::MAX, u64::MAX, 0, 0],
            ]) }};
        "
            )?;
        }
        write!(code, "}}\n")?;
        write!(
            code,
            "
        #[cfg(feature = \"nightly\")]
        pub mod avx512 {{\n"
        )?;

        {
            write!(
                code,
                "pub const MR_DIV_N: usize = 2; pub const NR: usize = 2; pub const N: usize = 4;"
            )?;
            write!(
                code,
                "const XOR_MASKS: [crate::x86::__m512; 4] = unsafe {{ core::mem::transmute([
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
                            need_mask: true,
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
                            mul_addsub: "crate::x86::subadd_pd",
                            mul_subadd: "_mm512_fmaddsub_pd",
                            xor: "_mm512_xor_si512",
                        };

                        write!(code, "{kernel}")?;
                    }
                }
            }

            write!(
                code,
                "pub static MICROKERNELS: [[[nano_gemm_core::MicroKernel<num_complex::Complex<f64>>; 2]; 2]; 17] = [\n"
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
        }
        write!(code, "}}\n")?;
        write!(code, "}}\n")?;

        Ok(code)
    }
}
