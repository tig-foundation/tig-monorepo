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
  imports: [
    ButtonModule,
    DialogModule,
    ProgressBarModule,
  ],
  templateUrl: './nav-bar.component.html',
  styleUrl: './nav-bar.component.scss',
  providers: [MessageService],
})
export class NavBarComponent {
  tigService = inject(TigApisService);
  router = inject(Router);

  value: number = 0;

  interval: any;

  constructor(private messageService: MessageService, private ngZone: NgZone) {
  }

  ngOnInit() {
    this.ngZone.runOutsideAngular(() => {
      this.interval = setInterval(() => {
        this.ngZone.run(() => {
          this.value = this.value + 1;
          if (this.value >= 60) {
            this.tigService.init();
            this.value = 0;
            this.messageService.add({
              severity: 'info',
              summary: 'Data Refreshed',
              detail: 'Process Completed',
            });
          } else {
          }
        });
      }, 2000);
    });
  }

  ngOnDestroy() {
    if (this.interval) {
      clearInterval(this.interval);
    }
  }
}
