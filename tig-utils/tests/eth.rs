#[cfg(all(feature = "web3", test))]
mod tests {
    #[test]
    fn test_recover_address_from_msg_and_sig() {
        assert_eq!(
            tig_utils::recover_address_from_msg_and_sig(
                "test message",
                "0x3556abec5b4b955f02a3643a5a3a29672caee32bbae180506452c84ab51c83f608dce8828923eccd1f3b81d8aca540864b917f1221b3a4d7833a4e3ec183369d1c"
            )
            .unwrap(),
            "0x7e6f4be25823b70c97009670da2eea36ce9096dc"
        );
    }

    #[tokio::test]
    async fn test_get_gnosis_safe_address() {
        assert_eq!(
            tig_utils::get_gnosis_safe_address(
                "https://mainnet.base.org",
                "0x0111ed72b3cd75e786083cf5d5db3a5ef0317891712315547fe49b3a16eebb16"
            )
            .await
            .unwrap(),
            "0x6ab6dffdee6efc0e60b34ef4685b40784c497af8".to_string()
        );
        assert_eq!(
            tig_utils::get_gnosis_safe_address(
                "https://mainnet.base.org",
                "0xbe95f21a4172d16cbf243ed6082e6c1cc132ec5ac3011a8b5289283f9059b8ce",
            )
            .await
            .unwrap(),
            "0xb787d8059689402dabc0a05832be526f79aa6b57".to_string()
        );
        assert_eq!(
            tig_utils::get_gnosis_safe_address(
                "https://sepolia.base.org",
                "0x90453a4f4ffffedeb70f144ce25d7ca74214f159ce394f0d929fda1d004d8ed0",
            )
            .await
            .unwrap(),
            "0x0112fb82f8041071d7000fb9e4782668e8cbd05f".to_string()
        );
    }

    #[tokio::test]
    async fn test_get_gnosis_safe_owners() {
        assert_eq!(
            tig_utils::get_gnosis_safe_owners(
                "https://mainnet.base.org",
                "0x6ab6dffdee6efc0e60b34ef4685b40784c497af8"
            )
            .await
            .unwrap(),
            vec!["0x7adc19694782c61132bcc38accaf1156d13c80d1".to_string()]
        );
        assert_eq!(
            tig_utils::get_gnosis_safe_owners(
                "https://mainnet.base.org",
                "0xb787d8059689402dabc0a05832be526f79aa6b57",
            )
            .await
            .unwrap(),
            vec![
                "0x9e10de6645d81823561aa4c91bef26c00b6c4d81".to_string(),
                "0xa327948a93000c6a37cac11b4239d7422a47c882".to_string(),
                "0x7a73bec3a56f935687dd30d0ff7c4bc632558c8b".to_string()
            ]
        );
        assert_eq!(
            tig_utils::get_gnosis_safe_owners(
                "https://sepolia.base.org",
                "0x0112fb82f8041071d7000fb9e4782668e8cbd05f",
            )
            .await
            .unwrap(),
            vec!["0x38d57e70513503c851f0a997fc1c8ab41cd2fca2".to_string()]
        );
    }

    #[tokio::test]
    async fn test_get_transaction() {
        assert_eq!(
            tig_utils::get_transaction(
                "https://mainnet.base.org",
                "0x0c03ce270b4826ec62e7dd007f0b716068639f7b",
                "0xb02b7c2a4bf72f7b5e96dcfa64d42cf75b765657998c9168beddcf06811b3701"
            )
            .await
            .unwrap(),
            tig_utils::Transaction {
                sender: "0x691108d12348ef0a153896492d96ff92bce90fe8".to_string(),
                receiver: "0x4e14297b4a5f7ab2e3c32ba262df2a2f8e367111".to_string(),
                amount: tig_utils::PreciseNumber::from_hex_str(
                    "00000000000000000000000000000000000000000000000178f3b8cd09d1681e"
                )
                .unwrap()
            }
        );
        assert_eq!(
            tig_utils::get_transaction(
                "https://sepolia.base.org",
                "0x3366feee9bbe5b830df9e1fa743828732b13959a",
                "0x093aa07701f2cb1ef62f0efcf101588898d6d2869edf66b8efc23969a15c218f",
            )
            .await
            .unwrap(),
            tig_utils::Transaction {
                sender: "0x26979f7282fc78cc83a74c5aeb317e7c13d33235".to_string(),
                receiver: "0xc30edf0147c46d0a5f79bfe9b15ce9de9b8879be".to_string(),
                amount: tig_utils::PreciseNumber::from(123)
            }
        );
    }
}
