include!(concat!(env!("OUT_DIR"), "/codegen.rs"));

#[cfg(test)]
mod tests {
    use std::arch::x86_64::__m256i;

    use super::*;

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

        let last_mask: __m256i = unsafe {
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

        unsafe {
            matmul_2_4_3(
                3,
                dst.as_mut_ptr() as *mut f32,
                15,
                a.as_ptr() as *const f32,
                17,
                b.as_ptr() as *const f32,
                2,
                6,
                beta,
                (&last_mask) as *const __m256i as *const (),
            )
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

        assert_eq!(dst, expected_dst);
    }
}
