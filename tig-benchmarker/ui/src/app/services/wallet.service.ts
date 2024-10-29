import { Injectable, signal } from '@angular/core';
import WalletLink from '@coinbase/wallet-sdk';
import { ethers } from 'ethers';
// import WalletConnect from '@walletconnect/client';
// import QRCodeModal from '@walletconnect/qrcode-modal';
@Injectable({
  providedIn: 'root',
})
export class WalletService {
  coinbaseWalletLink = new WalletLink({
    appName: 'TIG Benchmarker',
    appLogoUrl: 'https://tig.dev/favicon.ico',
  });
  coinbaseEthereum: any = null;
  provider: any = null;
  signer: any = null;
  wallet: any = signal(null);
  constructor() {}

  async connectCoinbaseWallet() {
    try {
      this.coinbaseEthereum = this.coinbaseWalletLink.makeWeb3Provider();
      console.log('Connecting to Coinbase Wallet...');
      this.provider = new ethers.BrowserProvider(this.coinbaseEthereum);

      // Request accounts from Coinbase Wallet
      const accounts = await this.provider.send('eth_requestAccounts', []);

      console.log('Connected accounts:', accounts);
      this.setWallet(accounts[0], 'coinbase');
      this.signer = this.provider.getSigner();

      // Get the user's account address
      const userAddress = await this.signer.getAddress();

      console.log('Connected account:', userAddress);
    } catch (error) {
      console.error('Error connecting to Coinbase Wallet:', error);
    }
  }

  setWallet(address: any, type: string) {
    const address_display =
      address.substring(0, 6) +
      '...' +
      address.substring(address.length - 4, address.length);
    this.wallet.set({
      address_display,
      address,
      type,
    });
  }

  // async connectWalletConnect() {
  //   try {
  //     // Check if connection is already established
  //     if (!this.walletConnector.connected) {
  //       // Create a new session
  //       this.walletConnector.createSession();
  //       this.walletConnector.on('connect', (error, payload) => {
  //         if (error) {
  //           throw error;
  //         }

  //         // Get the user's accounts and chainId from payload
  //         const { accounts, chainId } = payload.params[0];
  //         console.log('Connected account:', accounts[0], 'on chain', chainId);

  //         // Interact with the wallet using ethers.js
  //         // const provider = new ethers.BrowserProvider(this.walletConnector.);
  //         // const signer = provider.getSigner();

  //         // You can now send transactions, get balance, etc.
  //       });
  //     }
  //   } catch (error) {
  //     console.error('Error connecting to Coinbase Wallet:', error);
  //   }
  // }

  async connectMetaMask() {
    if (typeof (window as any).ethereum !== 'undefined') {
      console.log('MetaMask is installed!');

      try {
        const accounts = await (window as any).ethereum.request({
          method: 'eth_requestAccounts',
        });
        console.log('Connected account:', accounts[0]);
        this.setWallet(accounts[0], 'metamask');
        this.provider = new ethers.BrowserProvider((window as any).ethereum);
        this.signer = await this.provider.getSigner();

        const address = await this.signer.getAddress();

        console.log('User Address:', address);
      } catch (error) {
        console.error('User rejected the request:', error);
      }
    } else {
      console.error('MetaMask is not installed. Please install it!');
    }
  }

  disconnectWallet() {
    if (this.wallet()) {
      this.signer = null;
      this.provider = null;
      if (this.wallet().type === 'coinbase') {
        this.coinbaseEthereum.disconnect();
      }
      if (this.wallet().type === 'metamask') {
      }
      this.wallet.set(null);
    }
  }
}
