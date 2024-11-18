import { inject, Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';
import { TigApisService } from '../services/tig-apis.service';
import { WalletService } from '../services/wallet.service';

@Injectable({
  providedIn: 'root',
})
export class AuthGuard implements CanActivate {
  walletService = inject(WalletService);
  router = inject(Router);

  canActivate(): boolean {
    if (this.walletService.ready()) {
      return true; // Allow access if the user is authenticated
    } else {
      this.router.navigate(['/auth']); // Redirect to login if not authenticated
      return false; // Prevent access to the route
    }
  }
}
