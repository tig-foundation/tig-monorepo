// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {EIP712} from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract TopUpVault is EIP712, Pausable, Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    bytes32 public constant TOP_UP_AUTHORIZATION_TYPEHASH = keccak256(
        "TopUpAuthorization(address user,uint256 amount,bytes32 paymentId,uint256 deadline,uint256 chainId,address vault)"
    );

    IERC20 public immutable TIG;
    address public immutable BURN_ADDRESS;
    address public PAYMENT_AUTHORIZER;

    mapping(bytes32 paymentId => bool used) public usedPaymentIds;

    error InvalidAddress();
    error InvalidAmount();
    error AuthorizationExpired(uint256 deadline);
    error PaymentAlreadyUsed(bytes32 paymentId);
    error InvalidPaymentSignature(address recoveredSigner);

    event TopUpProcessed(
        address indexed user,
        bytes32 indexed paymentId,
        uint256 amount
    );
    event PaymentAuthorizerUpdated(
        address indexed previousAuthorizer,
        address indexed newAuthorizer
    );

    constructor(
        address tigToken,
        address burnAddress,
        address paymentAuthorizer
    ) EIP712("TopUpVault", "1") Ownable(msg.sender) {
        if (
            tigToken == address(0) ||
            burnAddress == address(0) ||
            paymentAuthorizer == address(0)
        ) {
            revert InvalidAddress();
        }

        TIG = IERC20(tigToken);
        BURN_ADDRESS = burnAddress;
        PAYMENT_AUTHORIZER = paymentAuthorizer;

        emit PaymentAuthorizerUpdated(address(0), paymentAuthorizer);
    }

    function executeTopUp(
        uint256 amount,
        bytes32 paymentId,
        uint256 deadline,
        bytes calldata signature
    ) external whenNotPaused nonReentrant {
        if (amount == 0) revert InvalidAmount();
        if (block.timestamp > deadline) revert AuthorizationExpired(deadline);
        if (usedPaymentIds[paymentId]) revert PaymentAlreadyUsed(paymentId);

        bytes32 structHash = keccak256(
            abi.encode(
                TOP_UP_AUTHORIZATION_TYPEHASH,
                msg.sender,
                amount,
                paymentId,
                deadline,
                block.chainid,
                address(this)
            )
        );
        address recoveredSigner = ECDSA.recover(
            _hashTypedDataV4(structHash),
            signature
        );
        if (recoveredSigner != PAYMENT_AUTHORIZER) {
            revert InvalidPaymentSignature(recoveredSigner);
        }

        usedPaymentIds[paymentId] = true;

        TIG.safeTransfer(msg.sender, amount);
        TIG.safeTransferFrom(msg.sender, BURN_ADDRESS, amount);

        emit TopUpProcessed(msg.sender, paymentId, amount);
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    function setPaymentAuthorizer(address newPaymentAuthorizer) external onlyOwner {
        if (newPaymentAuthorizer == address(0)) revert InvalidAddress();

        address previousAuthorizer = PAYMENT_AUTHORIZER;
        PAYMENT_AUTHORIZER = newPaymentAuthorizer;

        emit PaymentAuthorizerUpdated(previousAuthorizer, newPaymentAuthorizer);
    }

    function emergencyWithdrawTig(
        address recipient,
        uint256 amount
    ) external onlyOwner whenPaused nonReentrant {
        if (recipient == address(0)) revert InvalidAddress();
        TIG.safeTransfer(recipient, amount);
    }
}