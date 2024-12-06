import axios from 'axios';
import { inject, Injectable, signal } from '@angular/core';
import { toObservable } from '@angular/core/rxjs-interop';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';

@Injectable({
  providedIn: 'root',
})
export class TigApisService {
  base_url = 'http://localhost:3336';

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
    const url = `${this.base_url}/get-config`;
    const result = (await axios.get(url)).data;
    this.config.set(result);
    this.checkReady();
  }

  async saveConfig(data: any) {
    try {
      const url = `${this.base_url}/update-config`;
      const result = (await axios.post(url, data)).data;
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
    const url = `${this.base_url}/get-jobs`;
    const result = (await axios.get(url)).data;
    if (result) {
      this.benchmarks.set(
        result.map((b: any) => {
          let status = 'PENDING BENCHMARK';
          if (b.last_proof_submit_time) {
            status = 'SUBMITTED';
          } else if (b.last_benchmark_submit_time) {
            status = 'PENDING PROOF';
          }
          const num_solutions = b.solutions ? b.solutions.length : 0;

          let time_elapsed = 0;
          if (!b.last_proof_submit_time) {
            const start_timestamp = b.created_at;
            const start_normalizedTimestamp = start_timestamp.split('.')[0];
            const start_date = new Date(start_normalizedTimestamp);
            const seconds = Math.floor(start_date.getTime());
            const utcDate1 = new Date(Date.now());
            const utcDate2 = new Date(utcDate1.toUTCString());
            const now = utcDate2.getTime();
            time_elapsed = now - seconds;
          } else {
            const start_timestamp = b.created_at;
            const start_normalizedTimestamp = start_timestamp.split('.')[0];
            const start_date = new Date(start_normalizedTimestamp);
            const seconds = Math.floor(start_date.getTime());

            const end_timestamp = b.last_proof_submit_time;
            const end_normalizedTimestamp = end_timestamp.split('.')[0];
            const end_date = new Date(end_normalizedTimestamp);
            const end_seconds = Math.floor(end_date.getTime());
            time_elapsed = end_seconds - seconds;
          }

          const batches = b.batches.map((batch: any) => {
            const batch_number = batch.batch_number + 1;
            const num_solutions = batch.num_solutions;
            const status = batch.status;
            let time_elapsed = batch.elapsed_time;

            return {
              ...batch,
              batch_number,
              num_solutions,
              status,
              time_elapsed,
            };
          });

          return {
            ...b,
            status: status,
            num_solutions,
            time_elapsed,
            batches,
          };
        })
      );
    }
  }
}
