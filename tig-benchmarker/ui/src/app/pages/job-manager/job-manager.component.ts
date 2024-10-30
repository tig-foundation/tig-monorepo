import { Component, signal } from '@angular/core';
import { CardModule } from 'primeng/card';
import { TableModule } from 'primeng/table';
import { IconFieldModule } from 'primeng/iconfield';
import { InputIconModule } from 'primeng/inputicon';
import { InputTextModule } from 'primeng/inputtext';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { TagModule } from 'primeng/tag';
import { DatePipe, DecimalPipe } from '@angular/common';
import { ProgressBarModule } from 'primeng/progressbar';
@Component({
  selector: 'app-job-manager',
  standalone: true,
  imports: [
    CardModule,
    TableModule,
    TagModule,
    InputTextModule,
    InputIconModule,
    ProgressSpinnerModule,
    DatePipe,
    IconFieldModule,
    FormsModule,
    DecimalPipe,
    ReactiveFormsModule,
    ProgressBarModule,
  ],
  templateUrl: './job-manager.component.html',
  styleUrl: './job-manager.component.scss',
})
export class JobManagerComponent {
  jobs: any = signal(null);
  submissions: any = signal(null);
  constructor() {
    this.getJobs();
    this.getSubmissions();
  }

  ngOnInit(): void {}

  getJobs() {
    const jobs_test_data = [
      {
        start_time: '2021-05-22T08:23:45',
        challenge: 'Challenge 7',
        algorithm: 'Algorithm 33',
        total_batches: 500,
        batches_completed: 350,
        status: 'Running',
      },
      {
        start_time: '2023-04-11T14:05:12',
        challenge: 'Challenge 12',
        algorithm: 'Algorithm 9',
        total_batches: 200,
        batches_completed: 160,
        status: 'Completed',
      },
      {
        start_time: '2022-11-07T17:11:22',
        challenge: 'Challenge 5',
        algorithm: 'Algorithm 58',
        total_batches: 300,
        batches_completed: 100,
        status: 'Running',
      },
      {
        start_time: '2023-06-02T03:16:11',
        challenge: 'Challenge 19',
        algorithm: 'Algorithm 45',
        total_batches: 150,
        batches_completed: 120,
        status: 'Running',
      },
      {
        start_time: '2023-08-18T21:34:05',
        challenge: 'Challenge 23',
        algorithm: 'Algorithm 91',
        total_batches: 100,
        batches_completed: 50,
        status: 'Failed',
      },
      {
        start_time: '2021-12-25T10:12:32',
        challenge: 'Challenge 11',
        algorithm: 'Algorithm 72',
        total_batches: 600,
        batches_completed: 500,
        status: 'Completed',
      },
      {
        start_time: '2022-07-03T04:50:43',
        challenge: 'Challenge 8',
        algorithm: 'Algorithm 67',
        total_batches: 450,
        batches_completed: 220,
        status: 'Running',
      },
      {
        start_time: '2023-03-14T07:45:25',
        challenge: 'Challenge 16',
        algorithm: 'Algorithm 29',
        total_batches: 700,
        batches_completed: 630,
        status: 'Running',
      },
      {
        start_time: '2022-10-28T23:12:17',
        challenge: 'Challenge 3',
        algorithm: 'Algorithm 17',
        total_batches: 900,
        batches_completed: 750,
        status: 'Failed',
      },
      {
        start_time: '2021-01-30T15:26:40',
        challenge: 'Challenge 15',
        algorithm: 'Algorithm 4',
        total_batches: 120,
        batches_completed: 110,
        status: 'Completed',
      },
      {
        start_time: '2023-09-19T02:43:59',
        challenge: 'Challenge 10',
        algorithm: 'Algorithm 95',
        total_batches: 330,
        batches_completed: 250,
        status: 'Running',
      },
      {
        start_time: '2022-02-14T06:35:48',
        challenge: 'Challenge 14',
        algorithm: 'Algorithm 49',
        total_batches: 80,
        batches_completed: 70,
        status: 'Completed',
      },
      {
        start_time: '2021-09-30T19:53:31',
        challenge: 'Challenge 18',
        algorithm: 'Algorithm 84',
        total_batches: 900,
        batches_completed: 700,
        status: 'Running',
      },
      {
        start_time: '2023-05-06T09:17:52',
        challenge: 'Challenge 13',
        algorithm: 'Algorithm 51',
        total_batches: 670,
        batches_completed: 670,
        status: 'Completed',
      },
      {
        start_time: '2022-06-15T11:26:03',
        challenge: 'Challenge 20',
        algorithm: 'Algorithm 11',
        total_batches: 250,
        batches_completed: 200,
        status: 'Running',
      },
      {
        start_time: '2023-07-23T00:14:18',
        challenge: 'Challenge 6',
        algorithm: 'Algorithm 88',
        total_batches: 350,
        batches_completed: 200,
        status: 'Running',
      },
      {
        start_time: '2023-10-11T22:27:09',
        challenge: 'Challenge 22',
        algorithm: 'Algorithm 53',
        total_batches: 500,
        batches_completed: 100,
        status: 'Failed',
      },
      {
        start_time: '2023-02-09T20:08:56',
        challenge: 'Challenge 2',
        algorithm: 'Algorithm 77',
        total_batches: 600,
        batches_completed: 600,
        status: 'Completed',
      },
      {
        start_time: '2022-01-22T05:41:37',
        challenge: 'Challenge 9',
        algorithm: 'Algorithm 38',
        total_batches: 720,
        batches_completed: 720,
        status: 'Completed',
      },
      {
        start_time: '2023-11-01T13:50:29',
        challenge: 'Challenge 21',
        algorithm: 'Algorithm 80',
        total_batches: 270,
        batches_completed: 200,
        status: 'Running',
      },
    ];

    this.jobs.set(jobs_test_data);
  }

  getSubmissions(){
    const submissions_test_data = [
      {
        age: 32,
        solutions: 75,
        difficulty: '[245,755]',
        qualifiers: 12,
        imbalance: 1.68,
      },
      {
        age: 50,
        solutions: 15,
        difficulty: '[154,620]',
        qualifiers: 88,
        imbalance: 0.91,
      },
      {
        age: 94,
        solutions: 4,
        difficulty: '[49,897]',
        qualifiers: 21,
        imbalance: 1.14,
      },
      {
        age: 16,
        solutions: 42,
        difficulty: '[181,933]',
        qualifiers: 99,
        imbalance: 0.53,
      },
      {
        age: 67,
        solutions: 33,
        difficulty: '[215,705]',
        qualifiers: 70,
        imbalance: 1.85,
      },
      {
        age: 89,
        solutions: 95,
        difficulty: '[106,576]',
        qualifiers: 10,
        imbalance: 1.75,
      },
      {
        age: 47,
        solutions: 14,
        difficulty: '[367,782]',
        qualifiers: 3,
        imbalance: 1.57,
      },
      {
        age: 2,
        solutions: 78,
        difficulty: '[89,944]',
        qualifiers: 29,
        imbalance: 1.09,
      },
      {
        age: 34,
        solutions: 6,
        difficulty: '[173,511]',
        qualifiers: 25,
        imbalance: 0.8,
      },
      {
        age: 70,
        solutions: 56,
        difficulty: '[287,896]',
        qualifiers: 47,
        imbalance: 0.66,
      },
      {
        age: 91,
        solutions: 2,
        difficulty: '[176,569]',
        qualifiers: 40,
        imbalance: 0.33,
      },
      {
        age: 59,
        solutions: 87,
        difficulty: '[403,751]',
        qualifiers: 76,
        imbalance: 1.07,
      },
      {
        age: 38,
        solutions: 92,
        difficulty: '[131,684]',
        qualifiers: 58,
        imbalance: 1.9,
      },
      {
        age: 22,
        solutions: 13,
        difficulty: '[59,877]',
        qualifiers: 15,
        imbalance: 1.36,
      },
      {
        age: 11,
        solutions: 49,
        difficulty: '[255,753]',
        qualifiers: 91,
        imbalance: 0.95,
      },
      {
        age: 66,
        solutions: 66,
        difficulty: '[118,944]',
        qualifiers: 32,
        imbalance: 0.47,
      },
      {
        age: 45,
        solutions: 31,
        difficulty: '[71,994]',
        qualifiers: 90,
        imbalance: 1.8,
      },
      {
        age: 75,
        solutions: 52,
        difficulty: '[334,681]',
        qualifiers: 79,
        imbalance: 1.04,
      },
      {
        age: 29,
        solutions: 99,
        difficulty: '[214,547]',
        qualifiers: 86,
        imbalance: 1.93,
      },
      {
        age: 87,
        solutions: 34,
        difficulty: '[382,588]',
        qualifiers: 20,
        imbalance: 0.24,
      },
    ];

    this.submissions.set(submissions_test_data);
  }
}
