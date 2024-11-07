import axios from 'axios';
import BigNumber from 'bignumber.js';
import { inject, Injectable, signal } from '@angular/core';
import { toObservable } from '@angular/core/rxjs-interop';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';

@Injectable({
  providedIn: 'root',
})
export class TigApisService {
  base_url = 'http://localhost:3336';
  tig_url = 'https://testnet-api.tig.foundation'; // 'https://api.tig.foundation';

  // Services
  router = inject(Router);
  messageService = inject(MessageService);

  player_stats: any = signal(null);

  // Signals
  ready: any = signal(false);
  ready$ = toObservable(this.ready);
  algorithms: any = signal(null);
  challenges: any = signal(null);
  price_info: any = signal(null);
  price_info$ = toObservable(this.price_info);
  player_id: any = signal(null); // "0xda89f569b7f1a19236e37babe9960948cde2df23"
  api_key: any = signal(null); // "9349b6b6db9c11c53bc1b8ae71a7c53f"
  latest_block: any = signal(null);
  latest_block$ = toObservable(this.latest_block);
  config: any = signal(null);
  config$ = toObservable(this.config);

  batches: any = signal(null);
  benchmarks: any = signal(null);
  slaves: any = signal(null);
  constructor() {
    this.init();
  }

  async init() {
    const player_id = localStorage.getItem('player_id');
    const api_key = localStorage.getItem('api_key');
    if (player_id && api_key) {
      this.player_id.set(player_id);
      this.api_key.set(api_key);
      await this.initData();
      this.router.navigate(['/home']);
    } else {
      this.router.navigate(['/auth']);
    }
  }

  async setPlayerAndAuthKey(player_id: string, api_key: string) {
    this.player_id.set(player_id);
    this.api_key.set(api_key);
    localStorage.setItem('player_id', player_id);
    localStorage.setItem('api_key', api_key);
    const url = `${this.base_url}/register-player`;
    const result = (
      await axios.post(url, {
        player_id: player_id,
        api_key: api_key,
      })
    ).data;
    console.log('setPlayerAndAuthKey', result);

    this.init();
  }

  disconnect() {
    this.api_key.set(null);
    this.player_id.set(null);
    this.player_id.set(null);
    this.ready.set(false);
    this.algorithms.set(null);
    this.batches.set(null);
    this.benchmarks.set(null);
    localStorage.removeItem('player_id');
    localStorage.removeItem('api_key');
    this.router.navigate(['/auth']);
  }

  async initData() {
    this.getPriceInfo();
    await this.getLatestBlock();
    await this.getChallenges();
    await this.getAlgorithms();
    await this.getConfig();
    await this.getSlaves();
    await this.getBenchmarks();
    await this.getBatches();
  }

  async getAlgorithms() {
    if (this.latest_block()) {
      const url = `${this.tig_url}/get-algorithms?block_id=${
        this.latest_block()?.id
      }`;
      const result = (await axios.get(url)).data;
      if (result?.algorithms) {
        this.algorithms.set(
          result.algorithms.map((a: any) => {
            return {
              id: a.id,
              ...a.details,
              state: a.state,
              block_data: a.block_data,
            };
          })
        );
        console.log('algorithms', this.algorithms());
        this.checkReady();
      }
    }
  }
  async getChallenges() {
    console.log('getChallenges');
    if (this.latest_block()) {
      const temp_challenges = [
        {
          id: 'c001',
          name: 'satisfiability',
        },
        {
          id: 'c002',
          name: 'vehicle_routing',
        },
        {
          id: 'c003',
          name: 'knapsack',
        },
        {
          id: 'c004',
          name: 'vector_search',
        },
      ];
      console.log('challenges', temp_challenges);
      this.challenges.set(temp_challenges);
      this.checkReady();
    }
  }
  async getLatestBlock() {
    const url = `${this.base_url}/get-current-block`;
    const result = (
      await axios.get(url, {
        headers: {
          'x-api-key': this.api_key(),
        },
      })
    ).data;
    console.log('getLatestBlock', result);
    if (result?.player) {
      this.setCurrentPlayerStats(result.player);
    }
    if (result?.block) {
      this.latest_block.set(result.block);
    }
  }

  async setCurrentPlayerStats(player: any) {
    const data: any = {
      id: player.id,
      ...player.details,
      ...player.state,
      ...player.block_data,
    };

    data.deposit = new BigNumber(data.deposit || 0)
      .dividedBy(new BigNumber(Math.pow(10, 18)))
      .toString();
    data.total_fees_paid = new BigNumber(data.total_fees_paid || 0)
      .dividedBy(new BigNumber(Math.pow(10, 18)))
      .toString();
    data.round_earnings = new BigNumber(data.round_earnings || 0)
      .dividedBy(new BigNumber(Math.pow(10, 18)))
      .toString();
    data.rolling_deposit = new BigNumber(data.rolling_deposit || 0)
      .dividedBy(new BigNumber(Math.pow(10, 18)))
      .toString();
    data.available_fee_balance = new BigNumber(data.available_fee_balance || 0)
      .dividedBy(new BigNumber(Math.pow(10, 18)))
      .toString();
    data.reward = new BigNumber(data.reward || 0)
      .dividedBy(new BigNumber(Math.pow(10, 18)))
      .toString();
    this.player_stats.set(data);
    this.checkReady();
  }

  async getPriceInfo() {
    const url =
      'https://api.dexscreener.com/latest/dex/search?q=0x5280d5E63b416277d0F81FAe54Bb1e0444cAbDAA';
    const result = (await axios.get(url)).data;
    if (result.pairs) {
      const tig_data = result.pairs.filter(
        (p: any) => p.chainId === 'base' || p.baseToken.symbol === 'TIG'
      )[0];
      if (tig_data) {
        this.price_info.set(tig_data);
        this.checkReady();
      }
    }
  }

  checkReady() {
    if (
      this.algorithms() &&
      this.challenges() &&
      this.player_stats() &&
      this.config()
    ) {
      this.ready.set(true);
    }
  }

  async getConfig() {
    const url = `${this.base_url}/get-config`;
    const result = (
      await axios.get(url, {
        headers: {
          'x-api-key': this.api_key(),
        },
      })
    ).data;
    console.log('getConfig', result);
    this.config.set(result);
    this.checkReady();
  }

  async saveConfig(data: any) {
    console.log('saveConfig', data);
    try {
      const url = `${this.base_url}/update-config`;
      const result = (
        await axios.post(url, data, {
          headers: {
            'x-api-key': this.api_key(),
          },
        })
      ).data;
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

  async getSlaves() {
    if (this.latest_block()) {
      const url = `${this.base_url}/get-slaves?limit=1000&page=1`;
      const result = (
        await axios.get(url, {
          headers: {
            'x-api-key': this.api_key(),
          },
        })
      ).data;
      if (result?.slaves) {
        this.slaves.set(result.slaves);
        console.log('slaves', this.slaves());
      }
    }
  }
  async getBenchmarks() {
    if (this.latest_block()) {
      const url = `${this.base_url}/get-benchmark-jobs?limit=1000&page=1`;
      const result = (
        await axios.get(url, {
          headers: {
            'x-api-key': this.api_key(),
          },
        })
      ).data;
      if (result?.benchmark_jobs) {
        this.benchmarks.set(
          result.benchmark_jobs.map((b: any) => {
            // fraud
            // pending benchmark
            // pending proof
            // no solutions
            // active
            // active (qualifier)
            let status = 'PENDING BENCHMARK';
            if (b.completed_timestamp) {
              status = 'QUALIFIER';
            } else if (b.assigned_slave) {
              status = 'ACTIVE';
            } else if (b.completed_timestamp) {
              status = 'NO SOLUTIONS';
            } else if (b.pending_proof) {
              status = 'PENDING PROOF';
            } else if (b.pending_benchmark) {
              status = 'PENDING BENCHMARK';
            } else if (b.pending_benchmark) {
              status = 'FRAUD';
            }
            return {
              id: b.id,
              age: 'MISSING',
              challenge_id: b.settings.challenge_id,
              algorithm_id: b.settings.algorithm_id,
              start_time: b.start_time,
              end_time: b.end_time,
              status: status,
              solutions: b.solution_nonces.length,
              num_nonces: b.num_nonces,
              difficulty: b.settings.difficulty,
              // SET SUBMISSION DELAY HERE
              submission_delay: '',
              time_elapsed: 0,
            };
          })
        );
        console.log('benchmarks', this.benchmarks());
      }
    }
  }
  async getBatches() {
    if (this.latest_block()) {
      const url = `${this.base_url}/get-batches?limit=1000&page=1`;
      const result = (
        await axios.get(url, {
          headers: {
            'x-api-key': this.api_key(),
          },
        })
      ).data;
      if (result?.batches) {
        this.batches.set(result.batches);
        console.log('batches', this.batches());
      }
    }
  }
}
