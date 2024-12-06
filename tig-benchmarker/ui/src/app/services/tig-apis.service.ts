import axios from 'axios';
import { inject, Injectable, signal } from '@angular/core';
import { toObservable } from '@angular/core/rxjs-interop';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';

@Injectable({
  providedIn: 'root',
})
export class TigApisService {
  // Services
  router = inject(Router);
  messageService = inject(MessageService);

  // Signals
  ready: any = signal(false);
  ready$ = toObservable(this.ready);
  config: any = signal(null);
  config$ = toObservable(this.config);
  benchmarks: any = signal([]);
  benchmarks$ = toObservable(this.benchmarks);
  constructor() {
    this.init();
  }

  async init() {
    await this.getConfig();
    await this.getBenchmarks();
  }

  checkReady() {
    if (this.config()) {
      this.ready.set(true);
    }
  }

  async getConfig() {
    const result = (await axios.get('/get-config')).data;
    this.config.set(result);
    this.checkReady();
  }

  async saveConfig(data: any) {
    try {
      const result = (await axios.post('/update-config', data)).data;
      this.config.set(result);
      this.messageService.add({
        severity: 'success',
        summary: 'Success',
        detail: 'Config saved successfully',
        key: 'global-toast',
        life: 5000,
      });
      this.getConfig();
    } catch (e) {
      this.messageService.add({
        severity: 'error',
        summary: 'Error',
        detail: 'Error saving config',
        key: 'global-toast',
        life: 5000,
      });
      console.error('saveConfig', e);
    }
  }

  async getBenchmarks() {
    const result = (await axios.get('/get-jobs')).data;
    if (result) {
      this.benchmarks.set(
        result.map((b: any) => {
          const benchmark_id_display =
            b.benchmark_id.slice(0, 4) + '...' + b.benchmark_id.slice(-4);
          let time_elapsed = 0;
          if (!b.end_time) {
            const seconds = b.start_time;
            const utcDate1 = new Date(Date.now());
            const utcDate2 = new Date(utcDate1.toUTCString());
            const now = utcDate2.getTime();
            time_elapsed = now - seconds;
          } else {
            time_elapsed = b.end_time - b.start_time;
          }

          const batches = b.batches.map((batch: any) => {
            let time_elapsed = null;
            if (batch.start_time && !batch.end_time) {
              const seconds = batch.start_time;
              const utcDate1 = new Date(Date.now());
              const utcDate2 = new Date(utcDate1.toUTCString());
              const now = utcDate2.getTime();
              time_elapsed = now - seconds;
            } else if (batch.end_time && batch.start_time) {
              time_elapsed = batch.end_time - batch.start_time;
            }

            return {
              ...batch,
              time_elapsed,
            };
          });

          return {
            ...b,
            time_elapsed,
            benchmark_id_display,
            batches,
          };
        })
      );
    }
  }

  verifyBatch(batch: any) {
    console.log('verifyBatch', batch);
    const url = `/verify-batch/${batch.benchmark_id}_${batch.batch_number}`;
    axios.get(url).then(() => {
      this.getBenchmarks();
    });
  }
  stopBenchmark(benchmark: any) {
    console.log('benchmark', benchmark);
    const url = `/stop/${benchmark.benchmark_id}`;
    axios.get(url).then(() => {
      this.getBenchmarks();
    });
  }
}
