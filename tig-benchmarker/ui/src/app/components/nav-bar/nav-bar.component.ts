import { Component, inject, NgZone } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { DialogModule } from 'primeng/dialog';
import { Router } from '@angular/router';
import { TigApisService } from '../../services/tig-apis.service';
import { MessageService } from 'primeng/api';
import { ProgressBarModule } from 'primeng/progressbar';
@Component({
  selector: 'app-nav-bar',
  standalone: true,
  imports: [ButtonModule, DialogModule, ProgressBarModule],
  templateUrl: './nav-bar.component.html',
  styleUrl: './nav-bar.component.scss',
  providers: [MessageService],
})
export class NavBarComponent {
  tigService = inject(TigApisService);
  router = inject(Router);



  constructor() {}


}
