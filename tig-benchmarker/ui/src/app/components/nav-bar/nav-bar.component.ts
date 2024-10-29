import { Component, inject } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { WalletService } from '../../services/wallet.service';
import { DialogModule } from 'primeng/dialog';
import { WalletConnectorComponent } from '../wallet-connector/wallet-connector.component';
@Component({
  selector: 'app-nav-bar',
  standalone: true,
  imports: [ButtonModule,DialogModule,WalletConnectorComponent],
  templateUrl: './nav-bar.component.html',
  styleUrl: './nav-bar.component.scss',
})
export class NavBarComponent {
  wallet_service = inject(WalletService)
  wallet_connector_visible = false;
}
