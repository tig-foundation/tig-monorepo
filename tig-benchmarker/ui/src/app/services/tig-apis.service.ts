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
  algorithms: any = signal(null);
  challenges: any = signal(null);
  price_info: any = signal(null);
  price_info$ = toObservable(this.price_info);
  player_id: any = signal(null);
  api_key: any = signal(null); //"256979aea0c51f485e9559a45f29ec74"
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

  setPlayerAndAuthKey(player_id: string, api_key: string) {
    this.player_id.set(player_id);
    this.api_key.set(api_key);
    localStorage.setItem('player_id', player_id);
    localStorage.setItem('api_key', api_key);
    this.init();
  }

  async initData() {
    this.getPriceInfo();
    await this.getLatestBlock();
    await this.getChallenges();
    await this.getAlgorithms();
    await this.getConfig();
    await this.getSlaves();
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
    if (result?.block) {
      this.latest_block.set(result.block);
    }
    if (result?.player) {
      this.setCurrentPlayerStats(result.player);
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
      }
    }
  }
}
