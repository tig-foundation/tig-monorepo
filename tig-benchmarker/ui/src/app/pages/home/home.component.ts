import { Component, signal } from '@angular/core';
import { CardModule } from 'primeng/card';
import { TableModule } from 'primeng/table';
import { IconFieldModule } from 'primeng/iconfield';
import { InputIconModule } from 'primeng/inputicon';
import { InputTextModule } from 'primeng/inputtext';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { TagModule } from 'primeng/tag';
@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CardModule,
    TableModule,
    TagModule,
    InputTextModule,
    InputIconModule,
    ProgressSpinnerModule,
    IconFieldModule,
    FormsModule,
    ReactiveFormsModule,
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent {
  benchmarks: any = signal(null);
  solutions: any = signal(null);
  delegation: any = signal(null);

  constructor() {
    this.getBenchmarks();
    this.getSolutions();
  }

  getSolutions() {
    const solutions_test_data = [
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

    this.solutions.set(solutions_test_data);
  }
  getBenchmarks() {
    const benchmark_test_data = [
      {
        age: 5,
        challenge: 'Challenge 4',
        algorithm: 'Algorithm 77',
        solutions: 2,
        difficulty: '[37,880]',
        submission_delay: 1,
        status: 'Pending',
      },
      {
        age: 4,
        challenge: 'Challenge 73',
        algorithm: 'Algorithm 93',
        solutions: 47,
        difficulty: '[369,811]',
        submission_delay: 1,
        status: 'Pending',
      },
      {
        age: 4,
        challenge: 'Challenge 19',
        algorithm: 'Algorithm 79',
        solutions: 20,
        difficulty: '[446,565]',
        submission_delay: 1,
        status: 'In Progress',
      },
      {
        age: 4,
        challenge: 'Challenge 34',
        algorithm: 'Algorithm 31',
        solutions: 85,
        difficulty: '[358,844]',
        submission_delay: 1,
        status: 'Completed',
      },
      {
        age: 8,
        challenge: 'Challenge 19',
        algorithm: 'Algorithm 65',
        solutions: 82,
        difficulty: '[170,565]',
        submission_delay: 1,
        status: 'In Progress',
      },
      {
        age: 10,
        challenge: 'Challenge 19',
        algorithm: 'Algorithm 52',
        solutions: 45,
        difficulty: '[375,502]',
        submission_delay: 1,
        status: 'In Progress',
      },
      {
        age: 1,
        challenge: 'Challenge 3',
        algorithm: 'Algorithm 89',
        solutions: 56,
        difficulty: '[25,770]',
        submission_delay: 1,
        status: 'Completed',
      },
      {
        age: 4,
        challenge: 'Challenge 91',
        algorithm: 'Algorithm 2',
        solutions: 98,
        difficulty: '[376,672]',
        submission_delay: 1,
        status: 'In Progress',
      },
      {
        age: 1,
        challenge: 'Challenge 99',
        algorithm: 'Algorithm 98',
        solutions: 46,
        difficulty: '[467,547]',
        submission_delay: 1,
        status: 'Pending',
      },
      {
        age: 8,
        challenge: 'Challenge 83',
        algorithm: 'Algorithm 4',
        solutions: 25,
        difficulty: '[81,585]',
        submission_delay: 1,
        status: 'Pending',
      },
    ];
    this.benchmarks.set(benchmark_test_data);
  }
}
