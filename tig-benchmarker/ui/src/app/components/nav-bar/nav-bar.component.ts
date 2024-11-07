import { Component, inject, NgZone } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { WalletService } from '../../services/wallet.service';
import { DialogModule } from 'primeng/dialog';
import { WalletConnectorComponent } from '../wallet-connector/wallet-connector.component';
import { Router } from '@angular/router';
import { TigApisService } from '../../services/tig-apis.service';
import { MessageService } from 'primeng/api';
import { ProgressBarModule } from 'primeng/progressbar';
@Component({
  selector: 'app-nav-bar',
  standalone: true,
  imports: [
    ButtonModule,
    DialogModule,
    WalletConnectorComponent,
    ProgressBarModule,
  ],
  templateUrl: './nav-bar.component.html',
  styleUrl: './nav-bar.component.scss',
  providers: [MessageService],
})
export class NavBarComponent {
  wallet_service = inject(WalletService);
  tigService = inject(TigApisService);
  router = inject(Router);
  wallet_connector_visible = false;

  value: number = 0;

  interval: any;

  constructor(private messageService: MessageService, private ngZone: NgZone) {
    this.tigService.ready$.subscribe((ready: any) => {  
      console.log('ready observer', ready);
      if (ready) {
        this.wallet_service.ready.set(true);
      } else {
        this.wallet_service.ready.set(false);
      }
    }
    );
    // this.tigService.ready$.subscribe((ready: boolean) => {
    //   console.log('ready observer', ready);
    //   if (ready) {
    //     this.wallet_service.ready.set(true);
    //   } else {
    //     this.wallet_service.ready.set(false);
    //   }
    // });
  }

  ngOnInit() {
    this.ngZone.runOutsideAngular(() => {
      this.interval = setInterval(() => {
        this.ngZone.run(() => {
          this.value = this.value + 1;
          if (this.value >= 60) {
            this.tigService.initData();
            this.value = 0;
            this.messageService.add({
              severity: 'info',
              summary: 'Data Refreshed',
              detail: 'Process Completed',
            });
          } else {
          }
        });
      }, 5000);
    });
  }

  ngOnDestroy() {
    if (this.interval) {
      clearInterval(this.interval);
    }
  }
}
