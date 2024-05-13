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
                "0x2c84729e0c24ca982f598215a82888bf8620ebd1e42561b932ddf925b556ba51"
            )
            .await
            .unwrap(),
            tig_utils::Transaction {
                sender: "0x097d62f58e2986f2e1d85f259454af15f02f601e".to_string(),
                receiver: "0xe37bf84f75a8c3225d80aea2a95a24fd5a736895".to_string(),
                amount: tig_utils::PreciseNumber::from_dec_str("362450000000000000").unwrap()
            }
        );
        assert_eq!(
            tig_utils::get_transaction(
                "https://sepolia.base.org",
                "0x72589afe1f24794328f6ab0b43933ba0d2f8d7321d7bb6febe1513961dd3271e",
            )
            .await
            .unwrap(),
            tig_utils::Transaction {
                sender: "0xf0e1c3b2bf8e5ec5420b82d6361d074bc4d8b7f4".to_string(),
                receiver: "0xb88a1d716fa26dea1b20b2263c10ca2d7613e22f".to_string(),
                amount: tig_utils::PreciseNumber::from_dec_str("499990756092107808").unwrap()
            }
        );
    }
}
