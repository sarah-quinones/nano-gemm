#![no_std]

#[derive(Copy, Clone)]
pub struct MicroKernelData<T> {
    pub alpha: T,
    pub beta: T,
    pub conj_lhs: bool,
    pub conj_rhs: bool,
    pub k: usize,
    pub dst_cs: isize,
    pub lhs_cs: isize,
    pub rhs_rs: isize,
    pub rhs_cs: isize,
    pub last_mask: *const (),
}
unsafe impl<T: Sync> Sync for MicroKernelData<T> {}
unsafe impl<T: Send> Send for MicroKernelData<T> {}

pub type MicroKernel<T> =
    unsafe fn(data: &MicroKernelData<T>, dst: *mut T, lhs: *const T, rhs: *const T);
