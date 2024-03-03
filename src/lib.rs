#![cfg_attr(
    all(any(target_arch = "x86_64", target_arch = "x86"), feature = "nightly"),
    feature(stdarch_x86_avx512, avx512_target_feature)
)]

use core::mem::MaybeUninit;
use equator::debug_assert;

include!(concat!(env!("OUT_DIR"), "/codegen.rs"));

pub struct MicroKernelData<T> {
    pub alpha: T,
    pub beta: T,
    pub k: usize,
    pub dst_cs: isize,
    pub lhs_cs: isize,
    pub rhs_rs: isize,
    pub rhs_cs: isize,
    pub last_mask: *const (),
}

pub type MicroKernel<T> =
    unsafe fn(data: &MicroKernelData<T>, dst: *mut T, lhs: *const T, rhs: *const T);

#[derive(Debug, Copy, Clone)]
pub struct PlanReal<T> {
    microkernels: [[MaybeUninit<MicroKernel<T>>; 2]; 2],
    millikernel: unsafe fn(
        microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
        log_regsize: u32,
        mr: usize,
        nr: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut T,
        dst_rs: isize,
        dst_cs: isize,
        lhs: *const T,
        lhs_rs: isize,
        lhs_cs: isize,
        rhs: *const T,
        rhs_rs: isize,
        rhs_cs: isize,
        alpha: T,
        beta: T,
        full_mask: *const (),
        last_mask: *const (),
    ),
    log_regsize: u32,
    mr: usize,
    nr: usize,
    full_mask: *const (),
    last_mask: *const (),
    m: usize,
    n: usize,
    k: usize,
    dst_cs: isize,
    dst_rs: isize,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs_cs: isize,
    rhs_rs: isize,
}

#[allow(unused_variables)]
unsafe fn noop_millikernel<T: Copy>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
    log_regsize: u32,
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const T,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    rhs_cs: isize,
    alpha: T,
    beta: T,
    full_mask: *const (),
    last_mask: *const (),
) {
}

#[allow(unused_variables)]
unsafe fn naive_millikernel<
    T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + PartialEq,
>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
    log_regsize: u32,
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const T,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    rhs_cs: isize,
    alpha: T,
    beta: T,
    full_mask: *const (),
    last_mask: *const (),
) {
    let zero: T = core::mem::zeroed();
    if alpha == zero {
        for j in 0..n {
            for i in 0..m {
                let mut acc = zero;
                for depth in 0..k {
                    acc = acc
                        + *lhs.offset(lhs_rs * i as isize + lhs_cs * depth as isize)
                            * *rhs.offset(rhs_rs * depth as isize + rhs_cs * j as isize);
                }
                *dst.offset(dst_rs * i as isize + dst_cs * j as isize) = beta * acc;
            }
        }
    } else {
        for j in 0..n {
            for i in 0..m {
                let mut acc = zero;
                for depth in 0..k {
                    acc = acc
                        + *lhs.offset(lhs_rs * i as isize + lhs_cs * depth as isize)
                            * *rhs.offset(rhs_rs * depth as isize + rhs_cs * j as isize);
                }
                let dst = dst.offset(dst_rs * i as isize + dst_cs * j as isize);
                *dst = alpha * *dst + beta * acc;
            }
        }
    }
}

#[allow(unused_variables)]
unsafe fn fill_millikernel<T: Copy + PartialEq + core::ops::Mul<Output = T>>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
    log_regsize: u32,
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const T,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    rhs_cs: isize,
    alpha: T,
    beta: T,
    full_mask: *const (),
    last_mask: *const (),
) {
    let zero: T = core::mem::zeroed();
    if alpha == zero {
        for j in 0..n {
            for i in 0..m {
                *dst.offset(dst_rs * i as isize + dst_cs * j as isize) = core::mem::zeroed();
            }
        }
    } else {
        for j in 0..n {
            for i in 0..m {
                let dst = dst.offset(dst_rs * i as isize + dst_cs * j as isize);
                *dst = alpha * *dst;
            }
        }
    }
}

unsafe fn direct_millikernel<T: Copy>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
    log_regsize: u32,
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const T,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    rhs_cs: isize,
    alpha: T,
    beta: T,
    full_mask: *const (),
    last_mask: *const (),
) {
    debug_assert!(all(lhs_rs == 1, dst_rs == 1));
    _ = log_regsize;

    let mut data = MicroKernelData {
        alpha,
        beta,
        k,
        dst_cs,
        lhs_cs,
        rhs_rs,
        rhs_cs,
        last_mask,
    };

    let mut i = 0usize;
    while i < m {
        data.last_mask = if i + mr < m { full_mask } else { last_mask };
        let microkernels = microkernels.get_unchecked((i + mr >= m) as usize);
        let dst = dst.offset(i as isize);

        let mut j = 0usize;
        while j < n {
            let microkernel = microkernels
                .get_unchecked((j + nr >= n) as usize)
                .assume_init();

            microkernel(
                &data,
                dst.offset(j as isize * dst_cs),
                lhs.offset(i as isize),
                rhs.offset(j as isize * rhs_cs),
            );

            j += nr;
        }

        i += mr;
    }
}

trait One {
    const ONE: Self;
}

impl One for f32 {
    const ONE: Self = 1.0;
}
impl One for f64 {
    const ONE: Self = 1.0;
}

unsafe fn copy_millikernel<T: Copy + One>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
    log_regsize: u32,
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const T,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    rhs_cs: isize,
    mut alpha: T,
    beta: T,
    full_mask: *const (),
    last_mask: *const (),
) {
    if dst_rs == 1 && lhs_rs == 1 {
        let gemm_dst = dst;
        let gemm_lhs = lhs;
        let gemm_dst_cs = dst_cs;
        let gemm_lhs_cs = lhs_cs;

        direct_millikernel(
            microkernels,
            log_regsize,
            mr,
            nr,
            m,
            n,
            k,
            gemm_dst,
            1,
            gemm_dst_cs,
            gemm_lhs,
            1,
            gemm_lhs_cs,
            rhs,
            rhs_rs,
            rhs_cs,
            alpha,
            beta,
            full_mask,
            last_mask,
        );
    } else {
        let mut dst_tmp: MaybeUninit<[T; 32 * 32]> = core::mem::MaybeUninit::uninit();
        let mut lhs_tmp: MaybeUninit<[T; 32 * 32]> = core::mem::MaybeUninit::uninit();

        let dst_tmp = &mut *((&mut dst_tmp) as *mut _ as *mut [[MaybeUninit<T>; 32]; 32]);
        let lhs_tmp = &mut *((&mut lhs_tmp) as *mut _ as *mut [[MaybeUninit<T>; 32]; 32]);

        let gemm_dst_cs = 32;
        let gemm_lhs_cs = 32;

        let mut depth = 0usize;
        while depth < k {
            let depth_bs = Ord::min(32, k - depth);

            let mut i = 0usize;
            while i < m {
                let i_bs = Ord::min(32, m - i);

                let mut j = 0usize;
                while j < n {
                    let j_bs = Ord::min(32, n - j);

                    let gemm_dst = dst_tmp.as_mut_ptr() as *mut T;
                    let gemm_lhs = lhs_tmp.as_ptr() as *mut T;

                    let dst = dst.offset(dst_rs * i as isize + dst_cs * j as isize);
                    let lhs = lhs.offset(dst_rs * i as isize + dst_cs * j as isize);

                    for jj in 0..j_bs {
                        for ii in 0..i_bs {
                            *(gemm_dst.offset(ii as isize + gemm_dst_cs * jj as isize)
                                as *mut MaybeUninit<T>) = *(dst
                                .offset(dst_rs * ii as isize + dst_cs * jj as isize)
                                as *const MaybeUninit<T>);
                        }
                    }
                    for jj in 0..k {
                        for ii in 0..i_bs {
                            *(gemm_lhs.offset(ii as isize + gemm_lhs_cs * jj as isize)
                                as *mut MaybeUninit<T>) = *(lhs
                                .offset(lhs_rs * ii as isize + lhs_cs * jj as isize)
                                as *const MaybeUninit<T>);
                        }
                    }

                    direct_millikernel(
                        microkernels,
                        log_regsize,
                        mr,
                        nr,
                        m,
                        n,
                        k,
                        gemm_dst,
                        1,
                        gemm_dst_cs,
                        gemm_lhs,
                        1,
                        gemm_lhs_cs,
                        rhs,
                        rhs_rs,
                        rhs_cs,
                        alpha,
                        beta,
                        full_mask,
                        if i + i_bs == m { last_mask } else { full_mask },
                    );

                    for j in 0..n {
                        for i in 0..m {
                            *(dst.offset(dst_rs * i as isize + dst_cs * j as isize)
                                as *mut MaybeUninit<T>) = dst_tmp[j][i];
                        }
                    }

                    j += j_bs;
                }

                i += i_bs;
            }

            alpha = T::ONE;
            depth += depth_bs;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    use super::*;
    use equator::debug_assert;

    impl PlanReal<f32> {
        fn new_f32_scalar(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            Self {
                microkernels: [[MaybeUninit::<MicroKernel<f32>>::uninit(); 2]; 2],
                millikernel: naive_millikernel,
                log_regsize: 0,
                mr: 0,
                nr: 0,
                full_mask: core::ptr::null(),
                last_mask: core::ptr::null(),
                m,
                n,
                k,
                dst_rs: if is_col_major { 1 } else { isize::MIN },
                dst_cs: isize::MAX,
                lhs_rs: if is_col_major { 1 } else { isize::MIN },
                lhs_cs: isize::MAX,
                rhs_cs: isize::MAX,
                rhs_rs: isize::MAX,
            }
        }

        fn new_f32_avx(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            let mut microkernels = [[MaybeUninit::<MicroKernel<f32>>::uninit(); 2]; 2];

            let mr = 2 * 8;
            let nr = 4;

            {
                let k = Ord::min(k.wrapping_sub(1), 16);
                let m = (m.wrapping_sub(1) / 8) % (mr / 8);
                let n = n.wrapping_sub(1) % nr;

                microkernels[0][0].write(avx::f32::MICROKERNELS[k][1][3]);
                microkernels[0][1].write(avx::f32::MICROKERNELS[k][1][n]);
                microkernels[1][0].write(avx::f32::MICROKERNELS[k][m][3]);
                microkernels[1][1].write(avx::f32::MICROKERNELS[k][m][n]);
            }

            Self {
                microkernels,
                millikernel: if m == 0 || n == 0 {
                    noop_millikernel
                } else if k == 0 {
                    fill_millikernel
                } else if is_col_major {
                    direct_millikernel
                } else {
                    copy_millikernel
                },
                log_regsize: 8usize.ilog2(),
                mr,
                nr,
                m,
                n,
                k,
                dst_rs: if is_col_major { 1 } else { isize::MIN },
                dst_cs: isize::MIN,
                lhs_rs: if is_col_major { 1 } else { isize::MIN },
                lhs_cs: isize::MIN,
                rhs_cs: isize::MIN,
                rhs_rs: isize::MIN,
                full_mask: (&avx::f32::MASKS[0]) as *const _ as *const (),
                last_mask: (&avx::f32::MASKS[m % 8]) as *const _ as *const (),
            }
        }

        #[cfg(feature = "nightly")]
        fn new_f32_avx512(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            let mut microkernels = [[MaybeUninit::<MicroKernel<f32>>::uninit(); 2]; 2];

            let mr = 2 * 16;
            let nr = 4;

            {
                let k = Ord::min(k.wrapping_sub(1), 16);
                let m = (m.wrapping_sub(1) / 16) % (mr / 16);
                let n = n.wrapping_sub(1) % nr;

                microkernels[0][0].write(avx512::f32::MICROKERNELS[k][1][3]);
                microkernels[0][1].write(avx512::f32::MICROKERNELS[k][1][n]);
                microkernels[1][0].write(avx512::f32::MICROKERNELS[k][m][3]);
                microkernels[1][1].write(avx512::f32::MICROKERNELS[k][m][n]);
            }

            Self {
                microkernels,
                millikernel: if m == 0 || n == 0 {
                    noop_millikernel
                } else if k == 0 {
                    fill_millikernel
                } else if is_col_major {
                    direct_millikernel
                } else {
                    copy_millikernel
                },
                log_regsize: 16usize.ilog2(),
                mr,
                nr,
                m,
                n,
                k,
                dst_rs: if is_col_major { 1 } else { isize::MIN },
                dst_cs: isize::MIN,
                lhs_rs: if is_col_major { 1 } else { isize::MIN },
                lhs_cs: isize::MIN,
                rhs_cs: isize::MIN,
                rhs_rs: isize::MIN,
                full_mask: (&avx512::f32::MASKS[0]) as *const _ as *const (),
                last_mask: (&avx512::f32::MASKS[m % 16]) as *const _ as *const (),
            }
        }

        #[track_caller]
        pub fn new_colmajor_lhs_and_dst_f32(m: usize, n: usize, k: usize) -> Self {
            #[cfg(feature = "nightly")]
            if std::is_x86_feature_detected!("avx512f") {
                return Self::new_f32_avx512(m, n, k, true);
            }

            if std::is_x86_feature_detected!("avx")
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
            {
                return Self::new_f32_avx(m, n, k, true);
            }

            Self::new_f32_scalar(m, n, k, true)
        }

        #[track_caller]
        pub fn new_f32(m: usize, n: usize, k: usize) -> Self {
            #[cfg(feature = "nightly")]
            if std::is_x86_feature_detected!("avx512f") {
                return Self::new_f32_avx512(m, n, k, false);
            }

            if std::is_x86_feature_detected!("avx")
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
            {
                return Self::new_f32_avx(m, n, k, false);
            }

            Self::new_f32_scalar(m, n, k, false)
        }

        #[inline(always)]
        pub unsafe fn execute_unchecked(
            &self,
            m: usize,
            n: usize,
            k: usize,
            dst: *mut f32,
            dst_rs: isize,
            dst_cs: isize,
            lhs: *const f32,
            lhs_rs: isize,
            lhs_cs: isize,
            rhs: *const f32,
            rhs_rs: isize,
            rhs_cs: isize,
            alpha: f32,
            beta: f32,
        ) {
            debug_assert!(m == self.m);
            debug_assert!(n == self.n);
            debug_assert!(k == self.k);
            if self.dst_cs != isize::MIN {
                debug_assert!(dst_cs == self.dst_cs);
            }
            if self.dst_rs != isize::MIN {
                debug_assert!(dst_rs == self.dst_rs);
            }
            if self.lhs_cs != isize::MIN {
                debug_assert!(lhs_cs == self.lhs_cs);
            }
            if self.lhs_rs != isize::MIN {
                debug_assert!(lhs_rs == self.lhs_rs);
            }
            if self.rhs_cs != isize::MIN {
                debug_assert!(rhs_cs == self.rhs_cs);
            }
            if self.rhs_rs != isize::MIN {
                debug_assert!(rhs_rs == self.rhs_rs);
            }

            (self.millikernel)(
                &self.microkernels,
                self.log_regsize,
                self.mr,
                self.nr,
                m,
                n,
                k,
                dst,
                dst_rs,
                dst_cs,
                lhs,
                lhs_rs,
                lhs_cs,
                rhs,
                rhs_rs,
                rhs_cs,
                alpha,
                beta,
                self.full_mask,
                self.last_mask,
            );
        }

        #[inline(always)]
        pub unsafe fn execute_plan_free(
            m: usize,
            n: usize,
            k: usize,
            dst: *mut f32,
            dst_rs: isize,
            dst_cs: isize,
            lhs: *const f32,
            lhs_rs: isize,
            lhs_cs: isize,
            rhs: *const f32,
            rhs_rs: isize,
            rhs_cs: isize,
            alpha: f32,
            beta: f32,
        ) {
            Self::new_f32(m, n, k).execute_unchecked(
                m, n, k, dst, dst_rs, dst_cs, lhs, lhs_rs, lhs_cs, rhs, rhs_rs, rhs_cs, alpha, beta,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;

    #[test]
    fn test_kernel() {
        let gen = |_| rand::random::<f32>();
        let a: [[f32; 17]; 3] = core::array::from_fn(|_| core::array::from_fn(gen));
        let b: [[f32; 6]; 4] = core::array::from_fn(|_| core::array::from_fn(gen));
        let c: [[f32; 15]; 4] = core::array::from_fn(|_| core::array::from_fn(gen));
        assert!(std::is_x86_feature_detected!("avx"));
        assert!(std::is_x86_feature_detected!("avx2"));
        assert!(std::is_x86_feature_detected!("fma"));
        let mut dst = c;

        let last_mask: std::arch::x86_64::__m256i = unsafe {
            core::mem::transmute([
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                0,
            ])
        };

        let beta = 2.5;
        let alpha = 1.0;

        unsafe {
            avx::f32::matmul_2_4_dyn(
                &MicroKernelData {
                    alpha,
                    beta,
                    k: 3,
                    dst_cs: dst[0].len() as isize,
                    lhs_cs: a[0].len() as isize,
                    rhs_rs: 2,
                    rhs_cs: 6,
                    last_mask: (&last_mask) as *const _ as *const (),
                },
                dst.as_mut_ptr() as *mut f32,
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
            );
        };

        let mut expected_dst = c;
        for i in 0..15 {
            for j in 0..4 {
                let mut acc = 0.0f32;
                for depth in 0..3 {
                    acc = f32::mul_add(a[depth][i], b[j][2 * depth], acc);
                }
                expected_dst[j][i] = f32::mul_add(beta, acc, expected_dst[j][i]);
            }
        }

        assert!(dst == expected_dst);
    }

    #[test]
    fn test_plan() {
        let gen = |_| rand::random::<f32>();
        let m = 31;
        let n = 4;
        let k = 8;

        let a = (0..m * k).into_iter().map(gen).collect::<Vec<_>>();
        let b = (0..k * n).into_iter().map(gen).collect::<Vec<_>>();
        let c = (0..m * n).into_iter().map(|_| 0.0).collect::<Vec<_>>();
        let mut dst = c.clone();

        let plan = PlanReal::new_colmajor_lhs_and_dst_f32(m, n, k);
        let beta = 2.5;

        unsafe {
            plan.execute_unchecked(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                1,
                m as isize,
                a.as_ptr(),
                1,
                m as isize,
                b.as_ptr(),
                1,
                k as isize,
                1.0,
                beta,
            );
        };

        let mut expected_dst = c;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for depth in 0..k {
                    acc = f32::mul_add(a[depth * m + i], b[j * k + depth], acc);
                }
                expected_dst[j * m + i] = f32::mul_add(beta, acc, expected_dst[j * m + i]);
            }
        }

        assert!(dst == expected_dst);
    }

    #[test]
    fn test_plan_strided() {
        let gen = |_| rand::random::<f32>();
        let m = 31;
        let n = 4;
        let k = 8;

        let a = (0..2 * 33 * k).into_iter().map(gen).collect::<Vec<_>>();
        let b = (0..k * n).into_iter().map(gen).collect::<Vec<_>>();
        let c = (0..3 * 44 * n).into_iter().map(|_| 0.0).collect::<Vec<_>>();
        let mut dst = c.clone();

        let plan = PlanReal::new_f32(m, n, k);
        let beta = 2.5;

        unsafe {
            plan.execute_unchecked(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                3,
                44,
                a.as_ptr(),
                2,
                33,
                b.as_ptr(),
                1,
                k as isize,
                1.0,
                beta,
            );
        };

        let mut expected_dst = c;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for depth in 0..k {
                    acc = f32::mul_add(a[depth * 33 + i * 2], b[j * k + depth], acc);
                }
                expected_dst[j * 44 + i * 3] =
                    f32::mul_add(beta, acc, expected_dst[j * 44 + i * 3]);
            }
        }

        assert!(dst == expected_dst);
    }
}
