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
