// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract VestedRoundEarnings is Ownable, ReentrancyGuard {
    using SafeMath for uint256;

    IERC20 public token;
    uint256 public startTime;
    uint256 public constant VESTING_PERIOD = 7 days;
    uint256 public totalRoundEarnings;
    uint256 public totalAllocatedEarnings;

    bool public withdrawalsEnabled;

    struct Beneficiary {
        uint256 roundEarnings;
        uint256 withdrawnEarnings;
    }

    mapping(address => Beneficiary) public beneficiaries;

    event EarningsAdded(address beneficiary, uint256 amount);
    event TokensWithdrawn(address beneficiary, uint256 amount);
    event WithdrawalsEnabled();

    constructor(address _tokenAddress, uint256 _totalRoundEarnings) Ownable(msg.sender) {
        token = IERC20(_tokenAddress);
        totalRoundEarnings = _totalRoundEarnings;
        withdrawalsEnabled = false;
    }

    function addBeneficiaries(address[] memory _beneficiaries, uint256[] memory _earnings) external onlyOwner {
        require(!withdrawalsEnabled, "Withdrawals are already enabled");
        require(_beneficiaries.length == _earnings.length, "Arrays must have the same length");
        
        for (uint256 i = 0; i < _beneficiaries.length; i++) {
            address beneficiaryAddress = _beneficiaries[i];
            uint256 earnings = _earnings[i];
            
            require(beneficiaryAddress != address(0), "Invalid address");
            require(earnings > 0, "Earnings must be greater than 0");

            Beneficiary storage beneficiary = beneficiaries[beneficiaryAddress];
            beneficiary.roundEarnings = beneficiary.roundEarnings.add(earnings);
            totalAllocatedEarnings = totalAllocatedEarnings.add(earnings);

            emit EarningsAdded(beneficiaryAddress, earnings);
        }
    }

    function enableWithdrawals() external onlyOwner {
        require(!withdrawalsEnabled, "Withdrawals are already enabled");        
        require(totalAllocatedEarnings == totalRoundEarnings, "Total allocated earnings do not match totalRoundEarnings");
        require(token.balanceOf(address(this)) >= totalRoundEarnings, "Contract balance does not match totalRoundEarnings");
        
        startTime = block.timestamp;
        withdrawalsEnabled = true;
        
        emit WithdrawalsEnabled();
    }

    function withdraw() external nonReentrant {
        require(withdrawalsEnabled, "Withdrawals are not enabled yet");
        Beneficiary storage beneficiary = beneficiaries[msg.sender];
        require(beneficiary.roundEarnings > 0, "No earnings available");

        uint256 withdrawableAmount = getWithdrawableAmount(msg.sender);
        require(withdrawableAmount > 0, "No tokens available for withdrawal");

        beneficiary.withdrawnEarnings = beneficiary.withdrawnEarnings.add(withdrawableAmount);

        require(token.transfer(msg.sender, withdrawableAmount), "Withdrawal failed");

        emit TokensWithdrawn(msg.sender, withdrawableAmount);
    }

    function getWithdrawableAmount(address _beneficiary) public view returns (uint256) {
        require(withdrawalsEnabled, "Withdrawals are not enabled yet");
        Beneficiary storage beneficiary = beneficiaries[_beneficiary];
        uint256 elapsedTime = block.timestamp.sub(startTime);
        uint256 vestedAmount;
        if (elapsedTime >= VESTING_PERIOD) {
            vestedAmount = beneficiary.roundEarnings;
        } else {
            vestedAmount = beneficiary.roundEarnings.mul(elapsedTime).div(VESTING_PERIOD);
        }
        uint256 withdrawableAmount = vestedAmount.sub(beneficiary.withdrawnEarnings);
        return withdrawableAmount;
    }

    function getRoundEarnings(address _beneficiary) external view returns (uint256) {
        return beneficiaries[_beneficiary].roundEarnings;
    }

    function getWithdrawnEarnings(address _beneficiary) external view returns (uint256) {
        return beneficiaries[_beneficiary].withdrawnEarnings;
    }

    function getRemainingVestingPeriod() external view returns (uint256) {
        require(withdrawalsEnabled, "Withdrawals are not enabled yet");
        uint256 elapsedTime = block.timestamp.sub(startTime);
        if (elapsedTime >= VESTING_PERIOD) {
            return 0;
        }
        return VESTING_PERIOD.sub(elapsedTime);
    }
}