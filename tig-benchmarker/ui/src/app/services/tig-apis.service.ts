import axios from 'axios';
import BigNumber from 'bignumber.js';
import { inject, Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class TigApisService {
  player_stats : any = signal(null);
  algorithms : any = signal(null);
  challenges : any = signal(null);
  price_info : any = signal(null);
  round = 44;
  player_id = signal('0x81a8ed48a188853442e3ff2b5eab0c9fa9a3c626');
  latest_block: any = signal(null);
  constructor() {
    this.init();
  }

  async init() {
    await this.getLatestBlock();
    this.getAlgorithms();
    this.getChallenges();
    this.getCurrentPlayerStats();
    this.getPriceInfo();
  }

  async getLatestBlock() {
    const url =
      'https://mainnet-api.tig.foundation/get-block?round=42&include_data=true';
    const result = (await axios.get(url)).data;
    console.log('getLatestBlock', result);
    if (result?.block) {
      this.latest_block.set(result.block);
      console.log('latest_block', this.latest_block());
    }
  }
  async getAlgorithms() {
    if (this.latest_block()) {
      const url = `https://mainnet-api.tig.foundation/get-algorithms?block_id=${
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
      }
    }
  }
  async getChallenges() {
    console.log('getChallenges');
    if (this.latest_block()) {
      const url = `https://mainnet-api.tig.foundation/get-challenges?block_id=${
        this.latest_block()?.id
      }`;
      const result = (await axios.get(url)).data;
      console.log('getChallenges', result);
      // if (result?.algorithms) {
      //   this.algorithms.set(
      //     result.algorithms.map((a: any) => {
      //       return {
      //         id: a.id,
      //         ...a.details,
      //         state: a.state,
      //         block_data: a.block_data,
      //       };
      //     })
      //   );
      // }
    }
  }

  async getCurrentPlayerStats() {
    const url = `https://mainnet-api.tig.foundation/get-players?block_id=${
      this.latest_block()?.id
    }&player_type=benchmarker`;
    const result = (await axios.get(url)).data;
    if (result?.players) {
      if (result.players.filter((p: any) => p.id === this.player_id())[0]) {
        const player = result.players.filter(
          (p: any) => p.id === this.player_id()
        )[0];
        const data: any = {
          id: player.id,
          ...player.details,
          ...player.state,
          ...player.block_data,
        };

        data.deposit = new BigNumber(data.deposit)
          .dividedBy(new BigNumber(Math.pow(10, 18)))
          .toString();
        data.total_fees_paid = new BigNumber(data.total_fees_paid)
          .dividedBy(new BigNumber(Math.pow(10, 18)))
          .toString();
        data.round_earnings = new BigNumber(data.round_earnings)
          .dividedBy(new BigNumber(Math.pow(10, 18)))
          .toString();
        data.rolling_deposit = new BigNumber(data.rolling_deposit)
          .dividedBy(new BigNumber(Math.pow(10, 18)))
          .toString();
        data.available_fee_balance = new BigNumber(data.available_fee_balance)
          .dividedBy(new BigNumber(Math.pow(10, 18)))
          .toString();
        data.reward = new BigNumber(data.reward)
          .dividedBy(new BigNumber(Math.pow(10, 18)))
          .toString();
        this.player_stats.set(data);
        console.log('player_stats', this.player_stats());
      }
    }
  }

  async getPriceInfo() {
    const url =
      'https://api.dexscreener.com/latest/dex/search?q=0x5280d5E63b416277d0F81FAe54Bb1e0444cAbDAA';
    const result = (await axios.get(url)).data;
    if (result.pairs) {
      const tig_data = result.pairs.filter(
        (p: any) => p.chainId === 'base' || p.baseToken.symbol === 'TIG'
      )[0];
      console.log('price_info', tig_data);
      if (tig_data) {
        this.price_info.set(tig_data);
      }
    }
  }
}
