import { Component, inject, input, signal } from '@angular/core';
import {
  FormControl,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
} from '@angular/forms';
import { InputTextModule } from 'primeng/inputtext';
import { CardModule } from 'primeng/card';
import { InputNumberModule } from 'primeng/inputnumber';
import { ButtonModule } from 'primeng/button';
import { merge, tap } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';

@Component({
  selector: 'app-settings',
  standalone: true,

  imports: [
    CardModule,
    ButtonModule,
    ReactiveFormsModule,
    FormsModule,
    InputNumberModule,
    InputTextModule,
  ],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss',
})
export class SettingsComponent {
  settings: any = signal(null);
  formGroup!: FormGroup;
  view: any = input<string>('all');
  ngOnInit() {
    this.setCurrentConfig({});
  }

  setCurrentConfig(data: any) {
    console.log('data', data);
    this.formGroup = new FormGroup({
      // General
      api_key: new FormControl(data.api_key),
      player_id: new FormControl(data.player_id),
      ip_address: new FormControl(data.ip_address),
      port: new FormControl(data.port),
      precommits: new FormControl(data.precommits),

      // Batch Sizes
      satisfiability_batch_size: new FormControl(
        data.satisfiability_batch_size
      ),
      vehicle_routing_batch_size: new FormControl(
        data.vehicle_routing_batch_size
      ),
      knapsack_batch_size: new FormControl(data.knapsack_batch_size),
      vector_search_batch_size: new FormControl(data.vector_search_batch_size),
      // Algos
      satisfiability_algorithm: new FormControl(data.satisfiability_algorithm),
      vehicle_routing_algorithm: new FormControl(
        data.vehicle_routing_algorithm
      ),
      knapsack_algorithm: new FormControl(data.knapsack_algorithm),
      vector_search_algorithm: new FormControl(data.vector_search_algorithm),
      // Num Nonces
      satisfiability_num_nonces: new FormControl(
        data.satisfiability_num_nonces
      ),
      vehicle_routing_num_nonces: new FormControl(
        data.vehicle_routing_num_nonces
      ),
      knapsack_num_nonces: new FormControl(data.knapsack_num_nonces),
      vector_search_num_nonces: new FormControl(data.vector_search_num_nonces),
      // Weight
      satisfiability_weight: new FormControl(data.satisfiability_weight),
      vehicle_routing_weight: new FormControl(data.vehicle_routing_weight),
      knapsack_weight: new FormControl(data.knapsack_weight),
      vector_search_weight: new FormControl(data.vector_search_weight),

      // Difficulty
      satisfiability_difficulty: new FormControl(
        data.satisfiability_difficulty
      ),
      vehicle_routing_difficulty: new FormControl(
        data.vehicle_routing_difficulty
      ),
      knapsack_difficulty: new FormControl(data.knapsack_difficulty),
      vector_search_difficulty: new FormControl(data.vector_search_difficulty),

      // Micro Slave
      micro_slave_satisfiability: new FormControl(
        data.micro_slave_satisfiability
      ),
      micro_slave_vehicle_routing: new FormControl(
        data.micro_slave_vehicle_routing
      ),
      micro_slave_knapsack: new FormControl(data.micro_slave_knapsack),
      micro_slave_vector_search: new FormControl(
        data.micro_slave_vector_search
      ),
      // Small Slave
      small_slave_satisfiability: new FormControl(
        data.small_slave_satisfiability
      ),
      small_slave_vehicle_routing: new FormControl(
        data.small_slave_vehicle_routing
      ),
      small_slave_knapsack: new FormControl(data.small_slave_knapsack),
      small_slave_vector_search: new FormControl(
        data.small_slave_vector_search
      ),
      // Medium Slave
      medium_slave_satisfiability: new FormControl(
        data.medium_slave_satisfiability
      ),
      medium_slave_vehicle_routing: new FormControl(
        data.medium_slave_vehicle_routing
      ),
      medium_slave_knapsack: new FormControl(data.medium_slave_knapsack),
      medium_slave_vector_search: new FormControl(
        data.medium_slave_vector_search
      ),

      // Big Slave
      big_slave_satisfiability: new FormControl(data.big_slave_satisfiability),
      big_slave_vehicle_routing: new FormControl(
        data.big_slave_vehicle_routing
      ),
      big_slave_knapsack: new FormControl(data.big_slave_knapsack),
      big_slave_vector_search: new FormControl(data.big_slave_vector_search),
    });

    this.settings.set(data);
  }

  async saveForm() {
    console.log(this.formGroup.value);
  }
}
