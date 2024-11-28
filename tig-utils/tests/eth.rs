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
    async fn test_is_valid_gnosis_safe_sig() {
        assert!(
            tig_utils::is_valid_gnosis_safe_sig(
                "https://sepolia.base.org",
                "0x0C4ee0f015a47767E74AcC6F913d446867F2D571",
                "Test message...",
                "0xf1f2b72cdb594550c3c4f4d5825f092aed803d8182b0a2e9095fab81f08829e51aa52dbe8cd42d7a9e68ebb31578a4a3f761112c2c80e1ff455e348a904fb4061b"
            )
            .await.is_ok()
        );
        assert!(
            tig_utils::is_valid_gnosis_safe_sig(
                "https://sepolia.base.org",
                "0x0C4ee0f015a47767E74AcC6F913d446867F2D571",
                "EXPECT FAIL",
                "0xf1f2b72cdb594550c3c4f4d5825f092aed803d8182b0a2e9095fab81f08829e51aa52dbe8cd42d7a9e68ebb31578a4a3f761112c2c80e1ff455e348a904fb4061b"
            )
            .await
            .is_err()
        );
        assert!(
            tig_utils::is_valid_gnosis_safe_sig(
                "https://mainnet.base.org",
                "0x4664DF248d0cb2035316B41428013fe1cc48d054",
                "Test message...",
                "0x8ad0762136c7451850fce873b977008ac0c8b4e2397269c192f298b69362d38f2da89a3e7bc846f4f227f3311109c3ec1f11eef1257eef13a75588d5884183ec1b553d8c2a6a3df85ac002fb87efb7f61d229ab7eada526f12ba513f9d5addfb3c6cea4892c9cb5496dd9546202ec0e936782fe3805790b2724c057f210e6ae8651c"
            )
            .await
            .is_ok()
        );
        assert!(
            tig_utils::is_valid_gnosis_safe_sig(
                "https://mainnet.base.org",
                "0x4664DF248d0cb2035316B41428013fe1cc48d054",
                "EXPECT FAIL",
                "0x8ad0762136c7451850fce873b977008ac0c8b4e2397269c192f298b69362d38f2da89a3e7bc846f4f227f3311109c3ec1f11eef1257eef13a75588d5884183ec1b553d8c2a6a3df85ac002fb87efb7f61d229ab7eada526f12ba513f9d5addfb3c6cea4892c9cb5496dd9546202ec0e936782fe3805790b2724c057f210e6ae8651c"
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn test_get_transfer() {
        assert_eq!(
            tig_utils::get_transfer(
                "https://mainnet.base.org",
                "0xb02b7c2a4bf72f7b5e96dcfa64d42cf75b765657998c9168beddcf06811b3701",
                None
            )
            .await
            .unwrap(),
            tig_utils::Transfer {
                erc20: "0x0c03ce270b4826ec62e7dd007f0b716068639f7b".to_string(),
                sender: "0x691108d12348ef0a153896492d96ff92bce90fe8".to_string(),
                receiver: "0x4e14297b4a5f7ab2e3c32ba262df2a2f8e367111".to_string(),
                amount: tig_utils::PreciseNumber::from_hex_str(
                    "00000000000000000000000000000000000000000000000178f3b8cd09d1681e"
                )
                .unwrap(),
                log_idx: 0
            }
        );
        assert_eq!(
            tig_utils::get_transfer(
                "https://mainnet.base.org",
                "0xf68188c3913a45236a9435ac7b947448d607a2e7894c6dc205d45a3e3475dd9b",
                Some(5),
            )
            .await
            .unwrap(),
            tig_utils::Transfer {
                erc20: "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913".to_string(),
                sender: "0x6cdcb1c4a4d1c3c6d054b27ac5b77e89eafb971d".to_string(),
                receiver: "0x042c37762d1d126bc61eac2f5ceb7a96318f5db9".to_string(),
                amount: tig_utils::PreciseNumber::from_hex_str(
                    "00000000000000000000000000000000000000000000000000000000000dfec2"
                )
                .unwrap(),
                log_idx: 5
            }
        );
        assert_eq!(
            tig_utils::get_transfer(
                "https://sepolia.base.org",
                "0x093aa07701f2cb1ef62f0efcf101588898d6d2869edf66b8efc23969a15c218f",
                None,
            )
            .await
            .unwrap(),
            tig_utils::Transfer {
                erc20: "0x3366feee9bbe5b830df9e1fa743828732b13959a".to_string(),
                sender: "0x26979f7282fc78cc83a74c5aeb317e7c13d33235".to_string(),
                receiver: "0xc30edf0147c46d0a5f79bfe9b15ce9de9b8879be".to_string(),
                amount: tig_utils::PreciseNumber::from(123),
                log_idx: 0,
            }
        );
    }

    #[tokio::test]
    async fn test_get_linearlock() {
        assert_eq!(
            tig_utils::get_linear_lock(
                "https://mainnet.base.org",
                "0xcb183d3a0335eb71739dc58b926c68c0382d806cab9503b5bdeeb267eb961ea1",
                None
            )
            .await
            .unwrap(),
            tig_utils::LinearLock {
                locker: "0x4cb16d4153123a74bc724d161050959754f378d8".to_string(),
                erc20: "0x0c03ce270b4826ec62e7dd007f0b716068639f7b".to_string(),
                owner: "0xe716caba1aa085ad6f96d9027be4722a37a4e98a".to_string(),
                amount: tig_utils::PreciseNumber::from_hex_str(
                    "0000000000000000000000000000000000000000000008890c5abfd643a573f8"
                )
                .unwrap(),
                can_cancel: false,
                can_transfer: true,
                start_timestamp: 1728860015,
                cliff_timestamp: 0,
                end_timestamp: 1729464815,
                log_idx: 10
            }
        );
    }

    #[tokio::test]
    async fn test_lookup_ens_name() {
        assert_eq!(
            tig_utils::lookup_ens_name(
                "https://eth-mainnet.public.blastapi.io",
                "0x225f137127d9067788314bc7fcc1f36746a3c3B5",
            )
            .await
            .unwrap(),
            Some("luc.eth".to_string())
        );
        assert_eq!(
            tig_utils::lookup_ens_name(
                "https://eth-mainnet.public.blastapi.io",
                "0x0532d3971666953bc2db8619b89d546b3938a535",
            )
            .await
            .unwrap(),
            Some("harrisandtrotter.eth".to_string())
        );
        assert_eq!(
            tig_utils::lookup_ens_name(
                "https://eth-mainnet.public.blastapi.io",
                "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
            )
            .await
            .unwrap(),
            Some("vitalik.eth".to_string())
        );
    }
}
