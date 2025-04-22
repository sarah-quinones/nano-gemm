#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    all(any(target_arch = "x86_64", target_arch = "x86"), feature = "nightly"),
    feature(stdarch_x86_avx512, avx512_target_feature)
)]

use core::mem::MaybeUninit;
use equator::debug_assert;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod x86 {
    pub use nano_gemm_c32::x86::*;
    pub use nano_gemm_c64::x86::*;
    pub use nano_gemm_f32::x86::*;
    pub use nano_gemm_f64::x86::*;
}
#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    pub use nano_gemm_c32::aarch64::*;
    pub use nano_gemm_c64::aarch64::*;
    pub use nano_gemm_f32::aarch64::*;
    pub use nano_gemm_f64::aarch64::*;
}

#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex32;
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex64;

pub use nano_gemm_core::*;

#[derive(Copy, Clone)]
pub struct Plan<T> {
    microkernels: [[MaybeUninit<MicroKernel<T>>; 2]; 2],
    millikernel: unsafe fn(
        microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
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
        conj_lhs: bool,
        conj_rhs: bool,
        full_mask: *const (),
        last_mask: *const (),
    ),
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
    conj_lhs: bool,
    conj_rhs: bool,
    full_mask: *const (),
    last_mask: *const (),
) {
}

#[allow(unused_variables)]
unsafe fn naive_millikernel<
    T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + PartialEq + Conj,
>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
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
    conj_lhs: bool,
    conj_rhs: bool,
    full_mask: *const (),
    last_mask: *const (),
) {
    let zero: T = core::mem::zeroed();
    if alpha == zero {
        for j in 0..n {
            for i in 0..m {
                let mut acc = zero;
                for depth in 0..k {
                    let lhs = *lhs.offset(lhs_rs * i as isize + lhs_cs * depth as isize);
                    let rhs = *rhs.offset(rhs_rs * depth as isize + rhs_cs * j as isize);
                    acc = acc
                        + if conj_lhs { lhs.conj() } else { lhs }
                            * if conj_rhs { rhs.conj() } else { rhs };
                }
                *dst.offset(dst_rs * i as isize + dst_cs * j as isize) = beta * acc;
            }
        }
    } else {
        for j in 0..n {
            for i in 0..m {
                let mut acc = zero;
                for depth in 0..k {
                    let lhs = *lhs.offset(lhs_rs * i as isize + lhs_cs * depth as isize);
                    let rhs = *rhs.offset(rhs_rs * depth as isize + rhs_cs * j as isize);
                    acc = acc
                        + if conj_lhs { lhs.conj() } else { lhs }
                            * if conj_rhs { rhs.conj() } else { rhs };
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
    conj_lhs: bool,
    conj_rhs: bool,
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

#[inline(always)]
unsafe fn small_direct_millikernel<
    T: Copy,
    const M_DIVCEIL_MR: usize,
    const N_DIVCEIL_NR: usize,
>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
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
    conj_lhs: bool,
    conj_rhs: bool,
    full_mask: *const (),
    last_mask: *const (),
) {
    _ = (m, n);
    debug_assert!(all(lhs_rs == 1, dst_rs == 1));

    let mut data = MicroKernelData {
        alpha,
        beta,
        conj_lhs,
        conj_rhs,
        k,
        dst_cs,
        lhs_cs,
        rhs_rs,
        rhs_cs,
        last_mask,
    };

    let mut i = 0usize;
    while i < M_DIVCEIL_MR {
        data.last_mask = if i + 1 < M_DIVCEIL_MR {
            full_mask
        } else {
            last_mask
        };

        let microkernels = microkernels.get_unchecked((i + 1 >= M_DIVCEIL_MR) as usize);
        {
            let i = i * mr;
            let dst = dst.offset(i as isize);

            let mut j = 0usize;
            while j < N_DIVCEIL_NR {
                let microkernel = microkernels
                    .get_unchecked((j + 1 >= N_DIVCEIL_NR) as usize)
                    .assume_init();

                {
                    let j = j * nr;
                    microkernel(
                        &data,
                        dst.offset(j as isize * dst_cs),
                        lhs.offset(i as isize),
                        rhs.offset(j as isize * rhs_cs),
                    );
                }

                j += 1;
            }
        }
        i += 1;
    }
}

unsafe fn direct_millikernel<T: Copy>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
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
    conj_lhs: bool,
    conj_rhs: bool,
    full_mask: *const (),
    last_mask: *const (),
) {
    debug_assert!(all(lhs_rs == 1, dst_rs == 1));

    let mut data = MicroKernelData {
        alpha,
        beta,
        conj_lhs,
        conj_rhs,
        k,
        dst_cs,
        lhs_cs,
        rhs_rs,
        rhs_cs,
        last_mask,
    };

    let mut i = 0usize;
    while i < m {
        data.last_mask = if i + mr <= m { full_mask } else { last_mask };
        let microkernels = microkernels.get_unchecked((i + mr > m) as usize);
        let dst = dst.offset(i as isize);

        let mut j = 0usize;
        while j < n {
            let microkernel = microkernels
                .get_unchecked((j + nr > n) as usize)
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
trait Conj {
    fn conj(self) -> Self;
}

impl One for f32 {
    const ONE: Self = 1.0;
}
impl One for f64 {
    const ONE: Self = 1.0;
}
impl One for c32 {
    const ONE: Self = Self { re: 1.0, im: 0.0 };
}
impl One for c64 {
    const ONE: Self = Self { re: 1.0, im: 0.0 };
}

impl Conj for f32 {
    #[inline]
    fn conj(self) -> Self {
        self
    }
}
impl Conj for f64 {
    #[inline]
    fn conj(self) -> Self {
        self
    }
}

impl Conj for c32 {
    #[inline]
    fn conj(self) -> Self {
        Self::conj(&self)
    }
}
impl Conj for c64 {
    #[inline]
    fn conj(self) -> Self {
        Self::conj(&self)
    }
}

unsafe fn copy_millikernel<
    T: Copy + PartialEq + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Conj + One,
>(
    microkernels: &[[MaybeUninit<MicroKernel<T>>; 2]; 2],
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
    conj_lhs: bool,
    conj_rhs: bool,
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
            conj_lhs,
            conj_rhs,
            full_mask,
            last_mask,
        );
    } else {
        // 32 is always a multiple of both MR and NR
        const M_BS: usize = 64;
        const N_BS: usize = 64;
        const K_BS: usize = 64;
        let mut dst_tmp: MaybeUninit<[T; M_BS * N_BS]> = core::mem::MaybeUninit::uninit();
        let mut lhs_tmp: MaybeUninit<[T; M_BS * K_BS]> = core::mem::MaybeUninit::uninit();

        let dst_tmp = &mut *((&mut dst_tmp) as *mut _ as *mut [[MaybeUninit<T>; M_BS]; N_BS]);
        let lhs_tmp = &mut *((&mut lhs_tmp) as *mut _ as *mut [[MaybeUninit<T>; M_BS]; K_BS]);

        let gemm_dst = if dst_rs == 1 {
            dst
        } else {
            dst_tmp.as_mut_ptr() as *mut T
        };
        let gemm_lhs = lhs_tmp.as_mut_ptr() as *mut T;
        let gemm_dst_cs = if dst_rs == 1 { dst_cs } else { M_BS as isize };
        let gemm_lhs_cs = M_BS as isize;

        let mut depth = 0usize;
        while depth < k {
            let depth_bs = Ord::min(K_BS, k - depth);

            let mut i = 0usize;
            while i < m {
                let i_bs = Ord::min(M_BS, m - i);

                let lhs = lhs.offset(lhs_rs * i as isize + lhs_cs * depth as isize);

                for ii in 0..i_bs {
                    for jj in 0..depth_bs {
                        let ii = ii as isize;
                        let jj = jj as isize;
                        *(gemm_lhs.offset(ii + gemm_lhs_cs * jj)) =
                            *(lhs.offset(lhs_rs * ii + lhs_cs * jj));
                    }
                }

                let mut j = 0usize;
                while j < n {
                    let j_bs = Ord::min(N_BS, n - j);

                    let rhs = rhs.offset(rhs_rs * depth as isize + rhs_cs * j as isize);

                    let dst = dst.offset(dst_rs * i as isize + dst_cs * j as isize);
                    let gemm_dst = if dst_rs == 1 {
                        gemm_dst.offset(i as isize + gemm_dst_cs * j as isize)
                    } else {
                        gemm_dst
                    };

                    direct_millikernel(
                        microkernels,
                        mr,
                        nr,
                        i_bs,
                        j_bs,
                        depth_bs,
                        gemm_dst,
                        1,
                        gemm_dst_cs,
                        gemm_lhs,
                        1,
                        gemm_lhs_cs,
                        rhs,
                        rhs_rs,
                        rhs_cs,
                        if dst_rs == 1 {
                            alpha
                        } else {
                            core::mem::zeroed()
                        },
                        beta,
                        conj_lhs,
                        conj_rhs,
                        full_mask,
                        if i + i_bs == m { last_mask } else { full_mask },
                    );

                    if dst_rs != 1 {
                        if alpha == core::mem::zeroed() {
                            for ii in 0..i_bs {
                                for jj in 0..j_bs {
                                    let ii = ii as isize;
                                    let jj = jj as isize;
                                    *(dst.offset(dst_rs * ii + dst_cs * jj)) =
                                        *(gemm_dst.offset(ii + gemm_dst_cs * jj));
                                }
                            }
                        } else {
                            for ii in 0..i_bs {
                                for jj in 0..j_bs {
                                    let ii = ii as isize;
                                    let jj = jj as isize;
                                    let dst = dst.offset(dst_rs * ii + dst_cs * jj);
                                    *dst = alpha * *dst + *(gemm_dst.offset(ii + gemm_dst_cs * jj));
                                }
                            }
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

impl<T> Plan<T> {
    #[allow(dead_code)]
    #[inline(always)]
    fn from_masked_impl<const MR_DIV_N: usize, const NR: usize, const N: usize, Mask>(
        const_microkernels: &[[[MicroKernel<T>; NR]; MR_DIV_N]; 17],
        const_masks: Option<&[Mask; N]>,
        m: usize,
        n: usize,
        k: usize,
        is_col_major: bool,
    ) -> Self
    where
        T: Copy + PartialEq + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Conj + One,
    {
        let mut microkernels = [[MaybeUninit::<MicroKernel<T>>::uninit(); 2]; 2];

        let mr = MR_DIV_N * N;
        let nr = NR;

        {
            let k = Ord::min(k.wrapping_sub(1), 16);
            let m = (m.wrapping_sub(1) / N) % (mr / N);
            let n = n.wrapping_sub(1) % nr;

            microkernels[0][0].write(const_microkernels[k][MR_DIV_N - 1][NR - 1]);
            microkernels[0][1].write(const_microkernels[k][MR_DIV_N - 1][n]);
            microkernels[1][0].write(const_microkernels[k][m][NR - 1]);
            microkernels[1][1].write(const_microkernels[k][m][n]);
        }

        Self {
            microkernels,
            millikernel: if m == 0 || n == 0 {
                noop_millikernel
            } else if k == 0 {
                fill_millikernel
            } else if is_col_major {
                if m <= mr && n <= nr {
                    small_direct_millikernel::<_, 1, 1>
                } else if m <= mr && n <= 2 * nr {
                    small_direct_millikernel::<_, 1, 2>
                } else if m <= 2 * mr && n <= nr {
                    small_direct_millikernel::<_, 2, 1>
                } else if m <= 2 * mr && n <= 2 * nr {
                    small_direct_millikernel::<_, 2, 2>
                } else {
                    direct_millikernel
                }
            } else {
                copy_millikernel
            },
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
            full_mask: if let Some(const_masks) = const_masks {
                (&const_masks[0]) as *const _ as *const ()
            } else {
                &()
            },
            last_mask: if let Some(const_masks) = const_masks {
                (&const_masks[m % N]) as *const _ as *const ()
            } else {
                &()
            },
        }
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn from_non_masked_impl<const MR: usize, const NR: usize>(
        const_microkernels: &[[[MicroKernel<T>; NR]; MR]; 17],
        m: usize,
        n: usize,
        k: usize,
        is_col_major: bool,
    ) -> Self
    where
        T: Copy + PartialEq + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Conj + One,
    {
        let mut microkernels = [[MaybeUninit::<MicroKernel<T>>::uninit(); 2]; 2];

        let mr = MR;
        let nr = NR;

        {
            let k = Ord::min(k.wrapping_sub(1), 16);
            let m = m.wrapping_sub(1) % mr;
            let n = n.wrapping_sub(1) % nr;

            microkernels[0][0].write(const_microkernels[k][MR - 1][NR - 1]);
            microkernels[0][1].write(const_microkernels[k][MR - 1][n]);
            microkernels[1][0].write(const_microkernels[k][m][NR - 1]);
            microkernels[1][1].write(const_microkernels[k][m][n]);
        }

        Self {
            microkernels,
            millikernel: if m == 0 || n == 0 {
                noop_millikernel
            } else if k == 0 {
                fill_millikernel
            } else if is_col_major {
                if m <= mr && n <= nr {
                    small_direct_millikernel::<_, 1, 1>
                } else if m <= mr && n <= 2 * nr {
                    small_direct_millikernel::<_, 1, 2>
                } else if m <= 2 * mr && n <= nr {
                    small_direct_millikernel::<_, 2, 1>
                } else if m <= 2 * mr && n <= 2 * nr {
                    small_direct_millikernel::<_, 2, 2>
                } else {
                    direct_millikernel
                }
            } else {
                copy_millikernel
            },
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
            full_mask: &(),
            last_mask: &(),
        }
    }
}

impl Plan<f32> {
    fn new_f32_scalar(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        Self {
            microkernels: [[MaybeUninit::<MicroKernel<f32>>::uninit(); 2]; 2],
            millikernel: naive_millikernel,
            mr: 0,
            nr: 0,
            full_mask: core::ptr::null(),
            last_mask: core::ptr::null(),
            m,
            n,
            k,
            dst_rs: if is_col_major { 1 } else { isize::MIN },
            dst_cs: isize::MIN,
            lhs_rs: if is_col_major { 1 } else { isize::MIN },
            lhs_cs: isize::MIN,
            rhs_cs: isize::MIN,
            rhs_rs: isize::MIN,
        }
    }
}
impl Plan<f64> {
    fn new_f64_scalar(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        Self {
            microkernels: [[MaybeUninit::<MicroKernel<f64>>::uninit(); 2]; 2],
            millikernel: naive_millikernel,
            mr: 0,
            nr: 0,
            full_mask: core::ptr::null(),
            last_mask: core::ptr::null(),
            m,
            n,
            k,
            dst_rs: if is_col_major { 1 } else { isize::MIN },
            dst_cs: isize::MIN,
            lhs_rs: if is_col_major { 1 } else { isize::MIN },
            lhs_cs: isize::MIN,
            rhs_cs: isize::MIN,
            rhs_rs: isize::MIN,
        }
    }
}
impl Plan<c32> {
    fn new_c32_scalar(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        Self {
            microkernels: [[MaybeUninit::<MicroKernel<c32>>::uninit(); 2]; 2],
            millikernel: naive_millikernel,
            mr: 0,
            nr: 0,
            full_mask: core::ptr::null(),
            last_mask: core::ptr::null(),
            m,
            n,
            k,
            dst_rs: if is_col_major { 1 } else { isize::MIN },
            dst_cs: isize::MIN,
            lhs_rs: if is_col_major { 1 } else { isize::MIN },
            lhs_cs: isize::MIN,
            rhs_cs: isize::MIN,
            rhs_rs: isize::MIN,
        }
    }
}
impl Plan<c64> {
    fn new_c64_scalar(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        Self {
            microkernels: [[MaybeUninit::<MicroKernel<c64>>::uninit(); 2]; 2],
            millikernel: naive_millikernel,
            mr: 0,
            nr: 0,
            full_mask: core::ptr::null(),
            last_mask: core::ptr::null(),
            m,
            n,
            k,
            dst_rs: if is_col_major { 1 } else { isize::MIN },
            dst_cs: isize::MIN,
            lhs_rs: if is_col_major { 1 } else { isize::MIN },
            lhs_cs: isize::MIN,
            rhs_cs: isize::MIN,
            rhs_rs: isize::MIN,
        }
    }
}

impl<T> Plan<T> {
    #[inline(always)]
    pub unsafe fn execute_unchecked(
        &self,
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
        conj_lhs: bool,
        conj_rhs: bool,
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
            conj_lhs,
            conj_rhs,
            self.full_mask,
            self.last_mask,
        );
    }
}

impl Plan<f32> {
    #[track_caller]
    pub fn new_f32_impl(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        #[cfg(feature = "std")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly")]
            if m > 8 && std::is_x86_feature_detected!("avx512f") {
                return Self::new_f32_avx512(m, n, k, is_col_major);
            }

            if std::is_x86_feature_detected!("avx2") {
                if m == 1 {
                    return Self::new_f32x1(m, n, k, is_col_major);
                }
                if m == 2 {
                    return Self::new_f32x2(m, n, k, is_col_major);
                }
                if m <= 4 {
                    return Self::new_f32x4(m, n, k, is_col_major);
                }

                return Self::new_f32_avx(m, n, k, is_col_major);
            }
        }
        #[cfg(feature = "std")]
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::from_non_masked_impl(
                    &aarch64::f32::neon::MICROKERNELS,
                    m,
                    n,
                    k,
                    is_col_major,
                );
            }
        }

        Self::new_f32_scalar(m, n, k, is_col_major)
    }

    #[track_caller]
    pub fn new_colmajor_lhs_and_dst_f32(m: usize, n: usize, k: usize) -> Self {
        Self::new_f32_impl(m, n, k, true)
    }

    #[track_caller]
    pub fn new_f32(m: usize, n: usize, k: usize) -> Self {
        Self::new_f32_impl(m, n, k, false)
    }
}

impl Plan<f64> {
    #[track_caller]
    pub fn new_f64_impl(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        #[cfg(feature = "std")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly")]
            if m > 4 && std::is_x86_feature_detected!("avx512f") {
                return Self::new_f64_avx512(m, n, k, is_col_major);
            }

            if std::is_x86_feature_detected!("avx2") {
                if m == 1 {
                    return Self::new_f64x1(m, n, k, is_col_major);
                }
                if m == 2 {
                    return Self::new_f64x2(m, n, k, is_col_major);
                }

                return Self::new_f64_avx(m, n, k, is_col_major);
            }
        }

        #[cfg(feature = "std")]
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::from_non_masked_impl(
                    &aarch64::f64::neon::MICROKERNELS,
                    m,
                    n,
                    k,
                    is_col_major,
                );
            }
        }

        Self::new_f64_scalar(m, n, k, is_col_major)
    }

    #[track_caller]
    pub fn new_colmajor_lhs_and_dst_f64(m: usize, n: usize, k: usize) -> Self {
        Self::new_f64_impl(m, n, k, true)
    }

    #[track_caller]
    pub fn new_f64(m: usize, n: usize, k: usize) -> Self {
        Self::new_f64_impl(m, n, k, false)
    }
}

impl Plan<c32> {
    #[track_caller]
    pub fn new_c32_impl(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        #[cfg(feature = "std")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly")]
            if m > 4 && std::is_x86_feature_detected!("avx512f") {
                return Self::new_c32_avx512(m, n, k, is_col_major);
            }

            if std::is_x86_feature_detected!("avx2") {
                if m == 1 {
                    return Self::new_c32x1(m, n, k, is_col_major);
                }
                if m == 2 {
                    return Self::new_c32x2(m, n, k, is_col_major);
                }

                return Self::new_c32_avx(m, n, k, is_col_major);
            }
        }

        #[cfg(feature = "std")]
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon")
                && std::arch::is_aarch64_feature_detected!("fcma")
            {
                return Self::from_non_masked_impl(
                    &aarch64::c32::neon::MICROKERNELS,
                    m,
                    n,
                    k,
                    is_col_major,
                );
            }
        }

        Self::new_c32_scalar(m, n, k, is_col_major)
    }

    #[track_caller]
    pub fn new_colmajor_lhs_and_dst_c32(m: usize, n: usize, k: usize) -> Self {
        Self::new_c32_impl(m, n, k, true)
    }

    #[track_caller]
    pub fn new_c32(m: usize, n: usize, k: usize) -> Self {
        Self::new_c32_impl(m, n, k, false)
    }
}

impl Plan<c64> {
    #[track_caller]
    pub fn new_c64_impl(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
        #[cfg(feature = "std")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly")]
            if m > 2 && std::is_x86_feature_detected!("avx512f") {
                return Self::new_c64_avx512(m, n, k, is_col_major);
            }

            if std::is_x86_feature_detected!("avx2") {
                if m == 1 {
                    return Self::new_c64x1(m, n, k, is_col_major);
                }
                return Self::new_c64_avx(m, n, k, is_col_major);
            }
        }

        #[cfg(feature = "std")]
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon")
                && std::arch::is_aarch64_feature_detected!("fcma")
            {
                return Self::from_non_masked_impl(
                    &aarch64::c64::neon::MICROKERNELS,
                    m,
                    n,
                    k,
                    is_col_major,
                );
            }
        }
        Self::new_c64_scalar(m, n, k, is_col_major)
    }

    #[track_caller]
    pub fn new_colmajor_lhs_and_dst_c64(m: usize, n: usize, k: usize) -> Self {
        Self::new_c64_impl(m, n, k, true)
    }

    #[track_caller]
    pub fn new_c64(m: usize, n: usize, k: usize) -> Self {
        Self::new_c64_impl(m, n, k, false)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_api {
    use super::*;

    impl Plan<f32> {
        pub fn new_f32x1(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f32::f32x1::*;
            Self::from_masked_impl::<MR_DIV_N, NR, N, ()>(
                &MICROKERNELS,
                None,
                m,
                n,
                k,
                is_col_major,
            )
        }
        pub fn new_f32x2(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f32::f32x2::*;
            Self::from_masked_impl::<MR_DIV_N, NR, N, ()>(
                &MICROKERNELS,
                None,
                m,
                n,
                k,
                is_col_major,
            )
        }
        pub fn new_f32x4(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f32::f32x4::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }

        pub fn new_f32_avx(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f32::avx::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }

        #[cfg(feature = "nightly")]
        pub fn new_f32_avx512(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f32::avx512::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }
    }

    impl Plan<f64> {
        pub fn new_f64x1(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f64::f64x1::*;
            Self::from_masked_impl::<MR_DIV_N, NR, N, ()>(
                &MICROKERNELS,
                None,
                m,
                n,
                k,
                is_col_major,
            )
        }
        pub fn new_f64x2(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f64::f64x2::*;
            Self::from_masked_impl::<MR_DIV_N, NR, N, ()>(
                &MICROKERNELS,
                None,
                m,
                n,
                k,
                is_col_major,
            )
        }

        pub fn new_f64_avx(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f64::avx::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }

        #[cfg(feature = "nightly")]
        pub fn new_f64_avx512(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::f64::avx512::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }
    }
    impl Plan<c32> {
        pub fn new_c32x1(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::c32::c32x1::*;
            Self::from_masked_impl::<MR_DIV_N, NR, N, ()>(
                &MICROKERNELS,
                None,
                m,
                n,
                k,
                is_col_major,
            )
        }
        pub fn new_c32x2(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::c32::c32x2::*;
            Self::from_masked_impl::<MR_DIV_N, NR, N, ()>(
                &MICROKERNELS,
                None,
                m,
                n,
                k,
                is_col_major,
            )
        }

        pub fn new_c32_avx(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::c32::avx::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }

        #[cfg(feature = "nightly")]
        pub fn new_c32_avx512(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::c32::avx512::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }
    }
    impl Plan<c64> {
        pub fn new_c64x1(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::c64::c64x1::*;
            Self::from_masked_impl::<MR_DIV_N, NR, N, ()>(
                &MICROKERNELS,
                None,
                m,
                n,
                k,
                is_col_major,
            )
        }

        pub fn new_c64_avx(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::c64::avx::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }

        #[cfg(feature = "nightly")]
        pub fn new_c64_avx512(m: usize, n: usize, k: usize, is_col_major: bool) -> Self {
            use x86::c64::avx512::*;
            Self::from_masked_impl(&MICROKERNELS, Some(&MASKS), m, n, k, is_col_major)
        }
    }
}

pub mod planless {
    use super::*;

    #[inline(always)]
    pub unsafe fn execute_f32(
        mut m: usize,
        mut n: usize,
        k: usize,
        mut dst: *mut f32,
        mut dst_rs: isize,
        mut dst_cs: isize,
        mut lhs: *const f32,
        mut lhs_rs: isize,
        mut lhs_cs: isize,
        mut rhs: *const f32,
        mut rhs_rs: isize,
        mut rhs_cs: isize,
        alpha: f32,
        beta: f32,
        mut conj_lhs: bool,
        mut conj_rhs: bool,
    ) {
        if dst_cs.unsigned_abs() < dst_rs.unsigned_abs() {
            core::mem::swap(&mut m, &mut n);
            core::mem::swap(&mut dst_rs, &mut dst_cs);
            core::mem::swap(&mut lhs, &mut rhs);
            core::mem::swap(&mut lhs_rs, &mut rhs_cs);
            core::mem::swap(&mut lhs_cs, &mut rhs_rs);
            core::mem::swap(&mut conj_lhs, &mut conj_rhs);
        }
        if dst_rs == -1 && m > 0 {
            dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
            dst_rs = dst_rs.wrapping_neg();
            lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
            lhs_rs = lhs_rs.wrapping_neg();
        }

        let plan = if lhs_rs == 1 && dst_rs == 1 {
            Plan::new_colmajor_lhs_and_dst_f32(m, n, k)
        } else {
            Plan::new_f32(m, n, k)
        };
        plan.execute_unchecked(
            m, n, k, dst, dst_rs, dst_cs, lhs, lhs_rs, lhs_cs, rhs, rhs_rs, rhs_cs, alpha, beta,
            conj_lhs, conj_rhs,
        )
    }

    #[inline(always)]
    pub unsafe fn execute_c32(
        mut m: usize,
        mut n: usize,
        k: usize,
        mut dst: *mut c32,
        mut dst_rs: isize,
        mut dst_cs: isize,
        mut lhs: *const c32,
        mut lhs_rs: isize,
        mut lhs_cs: isize,
        mut rhs: *const c32,
        mut rhs_rs: isize,
        mut rhs_cs: isize,
        alpha: c32,
        beta: c32,
        mut conj_lhs: bool,
        mut conj_rhs: bool,
    ) {
        if dst_cs.unsigned_abs() < dst_rs.unsigned_abs() {
            core::mem::swap(&mut m, &mut n);
            core::mem::swap(&mut dst_rs, &mut dst_cs);
            core::mem::swap(&mut lhs, &mut rhs);
            core::mem::swap(&mut lhs_rs, &mut rhs_cs);
            core::mem::swap(&mut lhs_cs, &mut rhs_rs);
            core::mem::swap(&mut conj_lhs, &mut conj_rhs);
        }
        if dst_rs == -1 && m > 0 {
            dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
            dst_rs = dst_rs.wrapping_neg();
            lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
            lhs_rs = lhs_rs.wrapping_neg();
        }

        let plan = if lhs_rs == 1 && dst_rs == 1 {
            Plan::new_colmajor_lhs_and_dst_c32(m, n, k)
        } else {
            Plan::new_c32(m, n, k)
        };
        plan.execute_unchecked(
            m, n, k, dst, dst_rs, dst_cs, lhs, lhs_rs, lhs_cs, rhs, rhs_rs, rhs_cs, alpha, beta,
            conj_lhs, conj_rhs,
        )
    }

    #[inline(always)]
    pub unsafe fn execute_f64(
        mut m: usize,
        mut n: usize,
        k: usize,
        mut dst: *mut f64,
        mut dst_rs: isize,
        mut dst_cs: isize,
        mut lhs: *const f64,
        mut lhs_rs: isize,
        mut lhs_cs: isize,
        mut rhs: *const f64,
        mut rhs_rs: isize,
        mut rhs_cs: isize,
        alpha: f64,
        beta: f64,
        mut conj_lhs: bool,
        mut conj_rhs: bool,
    ) {
        if dst_cs.unsigned_abs() < dst_rs.unsigned_abs() {
            core::mem::swap(&mut m, &mut n);
            core::mem::swap(&mut dst_rs, &mut dst_cs);
            core::mem::swap(&mut lhs, &mut rhs);
            core::mem::swap(&mut lhs_rs, &mut rhs_cs);
            core::mem::swap(&mut lhs_cs, &mut rhs_rs);
            core::mem::swap(&mut conj_lhs, &mut conj_rhs);
        }
        if dst_rs == -1 && m > 0 {
            dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
            dst_rs = dst_rs.wrapping_neg();
            lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
            lhs_rs = lhs_rs.wrapping_neg();
        }

        let plan = if lhs_rs == 1 && dst_rs == 1 {
            Plan::new_colmajor_lhs_and_dst_f64(m, n, k)
        } else {
            Plan::new_f64(m, n, k)
        };
        plan.execute_unchecked(
            m, n, k, dst, dst_rs, dst_cs, lhs, lhs_rs, lhs_cs, rhs, rhs_rs, rhs_cs, alpha, beta,
            conj_lhs, conj_rhs,
        )
    }

    #[inline(always)]
    pub unsafe fn execute_c64(
        mut m: usize,
        mut n: usize,
        k: usize,
        mut dst: *mut c64,
        mut dst_rs: isize,
        mut dst_cs: isize,
        mut lhs: *const c64,
        mut lhs_rs: isize,
        mut lhs_cs: isize,
        mut rhs: *const c64,
        mut rhs_rs: isize,
        mut rhs_cs: isize,
        alpha: c64,
        beta: c64,
        mut conj_lhs: bool,
        mut conj_rhs: bool,
    ) {
        if dst_cs.unsigned_abs() < dst_rs.unsigned_abs() {
            core::mem::swap(&mut m, &mut n);
            core::mem::swap(&mut dst_rs, &mut dst_cs);
            core::mem::swap(&mut lhs, &mut rhs);
            core::mem::swap(&mut lhs_rs, &mut rhs_cs);
            core::mem::swap(&mut lhs_cs, &mut rhs_rs);
            core::mem::swap(&mut conj_lhs, &mut conj_rhs);
        }
        if dst_rs == -1 && m > 0 {
            dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
            dst_rs = dst_rs.wrapping_neg();
            lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
            lhs_rs = lhs_rs.wrapping_neg();
        }

        let plan = if lhs_rs == 1 && dst_rs == 1 {
            Plan::new_colmajor_lhs_and_dst_c64(m, n, k)
        } else {
            Plan::new_c64(m, n, k)
        };
        plan.execute_unchecked(
            m, n, k, dst, dst_rs, dst_cs, lhs, lhs_rs, lhs_cs, rhs, rhs_rs, rhs_cs, alpha, beta,
            conj_lhs, conj_rhs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_kernel() {
        let gen = |_| rand::random::<f32>();
        let a: [[f32; 17]; 3] = core::array::from_fn(|_| core::array::from_fn(gen));
        let b: [[f32; 6]; 4] = core::array::from_fn(|_| core::array::from_fn(gen));
        let c: [[f32; 15]; 4] = core::array::from_fn(|_| core::array::from_fn(gen));
        assert!(std::is_x86_feature_detected!("avx2"));
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
            x86::f32::avx::matmul_2_4_dyn(
                &MicroKernelData {
                    alpha,
                    beta,
                    conj_lhs: false,
                    conj_rhs: false,
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_kernel_cplx() {
        let gen = |_| rand::random::<c32>();
        let a: [[c32; 9]; 3] = core::array::from_fn(|_| core::array::from_fn(gen));
        let b: [[c32; 6]; 2] = core::array::from_fn(|_| core::array::from_fn(gen));
        let c: [[c32; 7]; 2] = core::array::from_fn(|_| core::array::from_fn(gen));
        assert!(std::is_x86_feature_detected!("avx2"));

        let last_mask: std::arch::x86_64::__m256i = unsafe {
            core::mem::transmute([
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                0,
                0,
            ])
        };

        let beta = c32::new(2.5, 3.5);
        let alpha = c32::new(1.0, 0.0);

        for (conj_lhs, conj_rhs) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut dst = c;
            unsafe {
                x86::c32::avx::matmul_2_2_dyn(
                    &MicroKernelData {
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        k: 3,
                        dst_cs: dst[0].len() as isize,
                        lhs_cs: a[0].len() as isize,
                        rhs_rs: 2,
                        rhs_cs: b[0].len() as isize,
                        last_mask: (&last_mask) as *const _ as *const (),
                    },
                    dst.as_mut_ptr() as *mut c32,
                    a.as_ptr() as *const c32,
                    b.as_ptr() as *const c32,
                );
            };

            let mut expected_dst = c;
            for i in 0..7 {
                for j in 0..2 {
                    let mut acc = c32::new(0.0, 0.0);
                    for depth in 0..3 {
                        let mut a = a[depth][i];
                        let mut b = b[j][2 * depth];
                        if conj_lhs {
                            a = a.conj();
                        }
                        if conj_rhs {
                            b = b.conj();
                        }
                        acc += a * b;
                    }
                    expected_dst[j][i] += beta * acc;
                }
            }

            for (&dst, &expected_dst) in
                core::iter::zip(dst.iter().flatten(), expected_dst.iter().flatten())
            {
                assert!((dst.re - expected_dst.re).abs() < 1e-5);
                assert!((dst.im - expected_dst.im).abs() < 1e-5);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_kernel_cplx64() {
        let gen = |_| rand::random::<c64>();
        let a: [[c64; 5]; 3] = core::array::from_fn(|_| core::array::from_fn(gen));
        let b: [[c64; 6]; 2] = core::array::from_fn(|_| core::array::from_fn(gen));
        let c: [[c64; 3]; 2] = core::array::from_fn(|_| core::array::from_fn(gen));
        assert!(std::is_x86_feature_detected!("avx2"));

        let last_mask: std::arch::x86_64::__m256i =
            unsafe { core::mem::transmute([u64::MAX, u64::MAX, 0, 0]) };

        let beta = c64::new(2.5, 3.5);
        let alpha = c64::new(1.0, 0.0);

        for (conj_lhs, conj_rhs) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut dst = c;
            unsafe {
                x86::c64::avx::matmul_2_2_dyn(
                    &MicroKernelData {
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        k: 3,
                        dst_cs: dst[0].len() as isize,
                        lhs_cs: a[0].len() as isize,
                        rhs_rs: 2,
                        rhs_cs: b[0].len() as isize,
                        last_mask: (&last_mask) as *const _ as *const (),
                    },
                    dst.as_mut_ptr() as *mut c64,
                    a.as_ptr() as *const c64,
                    b.as_ptr() as *const c64,
                );
            };

            let mut expected_dst = c;
            for i in 0..3 {
                for j in 0..2 {
                    let mut acc = c64::new(0.0, 0.0);
                    for depth in 0..3 {
                        let mut a = a[depth][i];
                        let mut b = b[j][2 * depth];
                        if conj_lhs {
                            a = a.conj();
                        }
                        if conj_rhs {
                            b = b.conj();
                        }
                        acc += a * b;
                    }
                    expected_dst[j][i] += beta * acc;
                }
            }

            for (&dst, &expected_dst) in
                core::iter::zip(dst.iter().flatten(), expected_dst.iter().flatten())
            {
                assert!((dst.re - expected_dst.re).abs() < 1e-5);
                assert!((dst.im - expected_dst.im).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn test_plan() {
        let gen = |_| rand::random::<f32>();
        for ((m, n), k) in (64..=64).zip(64..=64).zip([1, 4, 64]) {
            let a = (0..m * k).into_iter().map(gen).collect::<Vec<_>>();
            let b = (0..k * n).into_iter().map(gen).collect::<Vec<_>>();
            let c = (0..m * n).into_iter().map(|_| 0.0).collect::<Vec<_>>();
            let mut dst = c.clone();

            let plan = Plan::new_f32(m, n, k);
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
                    false,
                    false,
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

            for (dst, expected_dst) in dst.iter().zip(&expected_dst) {
                assert!((dst - expected_dst).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_plan_cplx() {
        let gen = |_| rand::random::<c64>();
        for ((m, n), k) in (0..128).zip(0..128).zip([1, 4, 17]) {
            let a = (0..m * k).into_iter().map(gen).collect::<Vec<_>>();
            let b = (0..k * n).into_iter().map(gen).collect::<Vec<_>>();
            let c = (0..m * n).into_iter().map(gen).collect::<Vec<_>>();

            for (conj_lhs, conj_rhs) in [(false, true), (false, false), (true, true), (true, false)]
            {
                for alpha in [c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(2.7, 3.7)] {
                    let mut dst = c.clone();

                    let plan = Plan::new_colmajor_lhs_and_dst_c64(m, n, k);
                    let beta = c64::new(2.5, 0.0);

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
                            alpha,
                            beta,
                            conj_lhs,
                            conj_rhs,
                        );
                    };

                    let mut expected_dst = c.clone();
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = c64::new(0.0, 0.0);
                            for depth in 0..k {
                                let mut a = a[depth * m + i];
                                let mut b = b[j * k + depth];
                                if conj_lhs {
                                    a = a.conj();
                                }
                                if conj_rhs {
                                    b = b.conj();
                                }
                                acc += a * b;
                            }
                            expected_dst[j * m + i] = alpha * expected_dst[j * m + i] + beta * acc;
                        }
                    }

                    for (&dst, &expected_dst) in core::iter::zip(dst.iter(), expected_dst.iter()) {
                        assert!((dst.re - expected_dst.re).abs() < 1e-5);
                        assert!((dst.im - expected_dst.im).abs() < 1e-5);
                    }
                }
            }
        }
    }

    #[test]
    fn test_plan_strided() {
        let gen = |_| rand::random::<f32>();
        for ((m, n), k) in (0..128).zip(0..128).zip([1, 4, 17]) {
            let a = (0..2 * 200 * k).into_iter().map(gen).collect::<Vec<_>>();
            let b = (0..k * n).into_iter().map(gen).collect::<Vec<_>>();
            let c = (0..3 * 400 * n)
                .into_iter()
                .map(|_| 0.0)
                .collect::<Vec<_>>();
            let mut dst = c.clone();

            let plan = Plan::new_f32(m, n, k);
            let beta = 2.5;

            unsafe {
                plan.execute_unchecked(
                    m,
                    n,
                    k,
                    dst.as_mut_ptr(),
                    3,
                    400,
                    a.as_ptr(),
                    2,
                    200,
                    b.as_ptr(),
                    1,
                    k as isize,
                    1.0,
                    beta,
                    false,
                    false,
                );
            };

            let mut expected_dst = c;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for depth in 0..k {
                        acc = f32::mul_add(a[depth * 200 + i * 2], b[j * k + depth], acc);
                    }
                    expected_dst[j * 400 + i * 3] =
                        f32::mul_add(beta, acc, expected_dst[j * 400 + i * 3]);
                }
            }

            for (dst, expected_dst) in dst.iter().zip(&expected_dst) {
                assert!((dst - expected_dst).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_plan_cplx_strided() {
        let gen = |_| c64::new(rand::random(), rand::random());
        for ((m, n), k) in (0..128).zip(0..128).zip([1, 4, 17, 190]) {
            let a = (0..2 * 200 * k).into_iter().map(gen).collect::<Vec<_>>();
            let b = (0..k * n).into_iter().map(gen).collect::<Vec<_>>();
            let c = (0..3 * 400 * n)
                .into_iter()
                .map(|_| c64::ZERO)
                .collect::<Vec<_>>();
            let mut dst = c.clone();

            let beta = 2.5.into();

            unsafe {
                planless::execute_c64(
                    m,
                    n,
                    k,
                    dst.as_mut_ptr(),
                    3,
                    400,
                    a.as_ptr(),
                    2,
                    200,
                    b.as_ptr(),
                    1,
                    k as isize,
                    1.0.into(),
                    beta,
                    false,
                    false,
                );
            };

            let mut expected_dst = c;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = c64::ZERO;
                    for depth in 0..k {
                        acc += a[depth * 200 + i * 2] * b[j * k + depth];
                    }
                    expected_dst[j * 400 + i * 3] = beta * acc + expected_dst[j * 400 + i * 3];
                }
            }

            for (dst, expected_dst) in dst.iter().zip(&expected_dst) {
                use num_complex::ComplexFloat;
                assert!((dst - expected_dst).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_plan_cplx_strided2() {
        let gen = |_| c64::new(rand::random(), rand::random());
        let m = 102;
        let n = 2;
        let k = 190;
        {
            let a = (0..2 * 200 * k).into_iter().map(gen).collect::<Vec<_>>();
            let b = (0..k * n).into_iter().map(gen).collect::<Vec<_>>();
            let c = (0..400 * n)
                .into_iter()
                .map(|_| c64::ZERO)
                .collect::<Vec<_>>();
            let mut dst = c.clone();

            let beta = 2.5.into();

            unsafe {
                planless::execute_c64(
                    m,
                    n,
                    k,
                    dst.as_mut_ptr(),
                    1,
                    400,
                    a.as_ptr(),
                    2,
                    200,
                    b.as_ptr(),
                    1,
                    k as isize,
                    1.0.into(),
                    beta,
                    false,
                    false,
                );
            };

            let mut expected_dst = c;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = c64::ZERO;
                    for depth in 0..k {
                        acc += a[depth * 200 + i * 2] * b[j * k + depth];
                    }
                    expected_dst[j * 400 + i] = beta * acc + expected_dst[j * 400 + i];
                }
            }

            for (dst, expected_dst) in dst.iter().zip(&expected_dst) {
                use num_complex::ComplexFloat;
                assert!((dst - expected_dst).abs() < 1e-4);
            }
        }
    }
}
