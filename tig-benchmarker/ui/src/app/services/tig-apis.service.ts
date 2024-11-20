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
    console.log('getConfig', result);
    this.config.set(result);
    this.checkReady();
  }

  async saveConfig(data: any) {
    console.log('saveConfig', data);
    try {
      const url = `${this.base_url}/update-config`;
      const result = (await axios.post(url, data)).data;
      console.log('saveConfig', result);
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
    console.log('getBenchmarks', url);
    const result = (await axios.get(url)).data;
    console.log('getBenchmarks', result);
    if (result) {
      this.benchmarks.set(
        result.map((b: any) => {
          let status = 'PENDING BENCHMARK';
           if (b.last_proof_submit_time) {
            status = 'SUBMITTED';
          } else if (b.last_benchmark_submit_time) {
            status = 'PENDING PROOF';
          } 
          return {
            ...b,
            status: status,
            time_elapsed: 0,
          };
        })
      );
      console.log('benchmarks', this.benchmarks());
    }
  }
}
