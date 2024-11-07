import { Component, inject } from '@angular/core';
import { BlockUIModule } from 'primeng/blockui';
import { ButtonModule } from 'primeng/button';
import { CheckboxModule } from 'primeng/checkbox';
import { WalletService } from '../../services/wallet.service';
import { FormsModule } from '@angular/forms';
import { ToastModule } from 'primeng/toast';
import { MessageService } from 'primeng/api';
@Component({
  selector: 'app-wallet-connector',
  standalone: true,
  imports: [
    ButtonModule,
    BlockUIModule,
    CheckboxModule,
    ToastModule,
    FormsModule,
  ],
  templateUrl: './wallet-connector.component.html',
  styleUrl: './wallet-connector.component.scss',

  providers: [MessageService],
})
export class WalletConnectorComponent {
  constructor(private messageService: MessageService) {}
  wallet_service = inject(WalletService);
  termsAccepted = false;

  connectWallet(type: string) {
    if (!this.termsAccepted) {
      this.messageService.add({
        severity: 'warn',
        summary: 'Error',
        detail: 'Please accept the terms and conditions',
      });
      return;
    }

    if (type === 'coinbase') {
      this.wallet_service.connectCoinbaseWallet();
    } else if (type === 'metamask') {
      this.wallet_service.connectMetaMask();
    } else if (type === 'wallet_connect') {
      // this.wallet_service.connectMetamaskWallet();
    }
  }


}
