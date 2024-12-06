import { Component, inject, NgZone, signal } from '@angular/core';
import { CardModule } from 'primeng/card';
import { TableModule } from 'primeng/table';
import { IconFieldModule } from 'primeng/iconfield';
import { InputIconModule } from 'primeng/inputicon';
import { InputTextModule } from 'primeng/inputtext';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { TagModule } from 'primeng/tag';
import { ButtonModule } from 'primeng/button';
import { TigApisService } from '../../services/tig-apis.service';
import { TabViewModule } from 'primeng/tabview';
import { ChartModule } from 'primeng/chart';
import { IBenchmark } from '../../interfaces/IBenchmark';
import { TimeConverterPipe } from '../../pipes/time-converter.pipe';
import { PanelModule } from 'primeng/panel';
import { DividerModule } from 'primeng/divider';
import { MessageService } from 'primeng/api';
import { ProgressBarModule } from 'primeng/progressbar';
@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CardModule,
    TableModule,
    TagModule,
    InputTextModule,
    ButtonModule,
    InputIconModule,
    PanelModule,
    ProgressSpinnerModule,
    DividerModule,
    IconFieldModule,
    FormsModule,
    ReactiveFormsModule,
    TabViewModule,
    ProgressBarModule,
    ChartModule,
    TimeConverterPipe,
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent {
  tigService = inject(TigApisService);

  // Benchmarks Table
  benchmarks: any = signal(null);
  expandedRows = {};

  constructor(private messageService: MessageService, private ngZone: NgZone) {
    this.init();
  }

  init() {
    this.tigService.benchmarks$.subscribe((data: any) => {
      this.getBenchmarks();
    });
  }

  getBenchmarks() {
    const benchmark_data: IBenchmark[] = this.tigService.benchmarks();
    this.benchmarks.set(benchmark_data);
  }

  expandAll() {
    this.expandedRows = this.benchmarks().reduce(
      (acc: any, p: any) => (acc[p.benchmark_id] = true) && acc,
      {}
    );
  }

  collapseAll() {
    this.expandedRows = {};
  }

  copyToClipboard(value: any) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard
        .writeText(value)
        .then(() => {
          console.log('Text copied to clipboard:', value);
        })
        .catch((err) => {
          console.error('Failed to copy text to clipboard:', err);
        });
    }
  }

  value: number = 0;
  timer: number = 0;
  interval: any;
  ngOnInit() {
    this.ngZone.runOutsideAngular(() => {
      this.interval = setInterval(() => {
        this.ngZone.run(() => {
          this.timer = this.timer + 1;
          this.value = Math.round((this.timer / 10) * 100);
          if (this.timer >= 10) {
            this.tigService.init();
            this.timer = 0;
            this.value = 0;
            this.messageService.add({
              severity: 'info',
              summary: 'Data Refreshed',
              detail: 'Process Completed',
            });
          } else {
          }
        });
      }, 1000);
    });
  }

  refreshTimer() {
    this.tigService.init();
    this.timer = 0;
    this.value = 0;
  }

  ngOnDestroy() {
    if (this.interval) {
      clearInterval(this.interval);
    }
  }
}
