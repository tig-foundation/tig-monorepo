import { Component, inject, signal } from '@angular/core';
import { CardModule } from 'primeng/card';
import { TableModule } from 'primeng/table';
import { IconFieldModule } from 'primeng/iconfield';
import { InputIconModule } from 'primeng/inputicon';
import { InputTextModule } from 'primeng/inputtext';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { TagModule } from 'primeng/tag';
import { ButtonModule } from 'primeng/button';
import { EditSettingsDialogComponent } from '../../components/edit-settings-dialog/edit-settings-dialog.component';
import { TigApisService } from '../../services/tig-apis.service';
import { TabViewModule } from 'primeng/tabview';
import { AsyncPipe, CurrencyPipe, DecimalPipe } from '@angular/common';
import { ChartModule } from 'primeng/chart';
import { IBenchmark } from '../../interfaces/IBenchmark';
import { AlgorithmPipe } from '../../pipes/algorithm.pipe';
import { ChallengePipe } from '../../pipes/challenge.pipe';
import { TimeConverterPipe } from '../../pipes/time-converter.pipe';
import { ICutoff } from '../../interfaces/ICutoff';
import { PanelModule } from 'primeng/panel';
import { IImbalance } from '../../interfaces/IImbalance';
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
    IconFieldModule,
    FormsModule,
    ReactiveFormsModule,
    EditSettingsDialogComponent,
    TabViewModule,
    CurrencyPipe,
    DecimalPipe,
    ChartModule,
    AsyncPipe,
    TimeConverterPipe,
    AlgorithmPipe,
    ChallengePipe,
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent {
  tigService = inject(TigApisService);
  price_usd: any = signal(null);
  volume_24h: any = signal(null);
  round_earnings: any = signal(null);
  latest_earnings: any = signal(null);
  available_fees: any = signal(null);

  // Rewards Graph
  rewards_data: any;
  rewards_labels: any;
  rewards_graph_options: any;
  // Benchmarks Table
  benchmarks: any = signal(null);

  // Cutoff and Imbalance
  cutoff: any = signal(null);
  imbalance: any = signal(null);

  // Challenges Table
  challenge_table: any = signal(null);
  challenge_summary: any = signal(null);

  //Delegation Cards
  locked_deposit: any = signal(null);
  total_delegated: any = signal(null);
  delegators: any = signal([]);

  constructor() {
    this.tigService.price_info$.subscribe((data: any) => {
      if (data) {
        this.price_usd.set(data.priceUsd);
        this.volume_24h.set(data.volume?.h24);
      }
    });
    // get Earnings and fees data
    this.tigService.latest_block$.subscribe((data: any) => {
      if (data) {
        this.available_fees.set(3.5);
        this.round_earnings.set(97.83);
        this.latest_earnings.set(1.5);
      }
    });
    this.getBenchmarks();
    this.initiateChart();
    this.getCutoffAndImbalance();
  }

  getBenchmarks() {
    const benchmark_test_data: IBenchmark[] = [
      {
        id: '1',
        age: 5,
        challenge_id: 'c001',
        algorithm_id: 'c001_a015',
        solutions: 2,
        difficulty: '[37,880]',
        submission_delay: 1,
        status: 'Pending',
        qualifiers: 1,
        number_of_nonces: 1,
        start_time: new Date().toISOString(),
        end_time: new Date(
          new Date().setMinutes(new Date().getMinutes() + 10)
        ).toISOString(),
      },
    ];

    benchmark_test_data.map((b) => {
      b.time_elapsed =
        new Date(b.end_time).getTime() - new Date(b.start_time).getTime();
    });
    this.benchmarks.set(benchmark_test_data);
  }

  initiateChart() {
    //get rewards data
    this.rewards_labels = [
      '1',
      '2',
      '3',
      '4',
      '5',
      '6',
      '7',
      '8',
      '9',
      '10',
      '11',
      '12',
      '13',
      '14',
      '15',
      '16',
      '17',
      '18',
      '19',
      '20',
      '21',
      '22',
      '23',
      '24',
      '25',
    ];
    const rewards_data = Array.from(
      { length: 25 },
      () => Math.floor(Math.random() * 550) / 100
    );

    // set chart data
    const documentStyle = getComputedStyle(document.documentElement);
    const textColor = documentStyle.getPropertyValue('--text-color');
    const textColorSecondary = documentStyle.getPropertyValue(
      '--text-color-secondary'
    );
    const surfaceBorder = documentStyle.getPropertyValue('--surface-border');

    this.rewards_data = {
      labels: this.rewards_labels,
      datasets: [
        {
          label: 'Block Rewards',
          data: rewards_data,
          fill: false,
          tension: 0.4,
        },
      ],
    };

    this.rewards_graph_options = {
      maintainAspectRatio: false,
      aspectRatio: 0.6,
      plugins: {
        legend: {
          labels: {
            color: textColor,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: textColorSecondary,
          },
          grid: {
            color: surfaceBorder,
            drawBorder: false,
          },
        },
        y: {
          ticks: {
            color: textColorSecondary,
          },
          grid: {
            color: surfaceBorder,
            drawBorder: false,
          },
        },
      },
    };
  }

  getCutoffAndImbalance() {
    const cutoff: ICutoff = {
      cutoff: 0.5,
      c001_qualifiers: 1,
      c002_qualifiers: 2,
      c003_qualifiers: 3,
      c004_qualifiers: 4,
      deposit: 100,
    };

    this.cutoff.set(cutoff);

    const imbalance: IImbalance = {
      imbalance: 25,
      c001_qualifiers: 1,
      c002_qualifiers: 2,
      c003_qualifiers: 3,
      c004_qualifiers: 4,
      c001_solutions: 10,
      c002_solutions: 20,
      c003_solutions: 30,
      c004_solutions: 40,
      deposit: 100,
      deposit_qualifiers: 50,
    };

    this.imbalance.set(imbalance);
  }

  changeChallengeView(challenge: any) {
    const selected = this.tigService.challenges()[challenge];
    this.challenge_table.set(
      this.benchmarks().filter((b: any) => b.challenge_id === selected.id)
    );
    // Set Challenge Summary Information
    const summary = {
      base_fee: 3,
      solutions: 16,
      qualifiers: 2,
    };
    this.challenge_summary.set(summary);
  }
}
