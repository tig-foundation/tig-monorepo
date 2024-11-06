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
import { TigApisService } from '../../services/tig-apis.service';
import { SliderModule } from 'primeng/slider';
import { DropdownModule } from 'primeng/dropdown';
@Component({
  selector: 'app-settings',
  standalone: true,

  imports: [
    CardModule,
    ButtonModule,
    ReactiveFormsModule,
    FormsModule,
    SliderModule,
    DropdownModule,
    InputNumberModule,
    InputTextModule,
  ],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss',
})
export class SettingsComponent {
  tigService: any = inject(TigApisService);
  settings: any = signal(null);
  formGroup!: FormGroup;
  view: any = input<string>('all');
  satisfiability_algos: any = this.tigService
    .algorithms()
    .filter((a: any) => a.challenge_id === 'c001');
  vehicle_routing_algos: any = this.tigService
    .algorithms()
    .filter((a: any) => a.challenge_id === 'c002');
  vector_search_algos: any = this.tigService
    .algorithms()
    .filter((a: any) => a.challenge_id === 'c004');
  knapsack_algos: any = this.tigService
    .algorithms()
    .filter((a: any) => a.challenge_id === 'c003');

  async ngOnInit() {
    this.tigService.config$.subscribe((data: any) => {
      if (data) {
        this.setCurrentConfig(data);
      }
    });
  }

  setCurrentConfig(data: any) {
    const satisfiability_algorithm = this.tigService
      .algorithms()
      .find(
        (a: any) =>
          a.name ===
          data.precommit_manager_config.algo_selection.satisfiability.algorithm
      );
    const knapsack_algorithm = this.tigService
      .algorithms()
      .find(
        (a: any) =>
          a.name ===
          data.precommit_manager_config.algo_selection.knapsack.algorithm
      );
    const vehicle_routing_algorithm = this.tigService
      .algorithms()
      .find(
        (a: any) =>
          a.name ===
          data.precommit_manager_config.algo_selection.vehicle_routing.algorithm
      );
    const vector_search_algorithm = this.tigService
      .algorithms()
      .find(
        (a: any) =>
          a.name ===
          data.precommit_manager_config.algo_selection.vector_search.algorithm
      );
    this.formGroup = new FormGroup({
      max_pending_benchmarks: new FormControl(
        data.precommit_manager_config.max_pending_benchmarks
      ),
      // Batch Sizes
      satisfiability_batch_size: new FormControl(
        data.precommit_manager_config.algo_selection.satisfiability.num_nonces
      ),
      vehicle_routing_batch_size: new FormControl(
        data.precommit_manager_config.algo_selection.vehicle_routing.num_nonces
      ),
      knapsack_batch_size: new FormControl(
        data.precommit_manager_config.algo_selection.knapsack.num_nonces
      ),
      vector_search_batch_size: new FormControl(
        data.precommit_manager_config.algo_selection.vector_search.num_nonces
      ),
      // Algos

      satisfiability_algorithm: new FormControl(satisfiability_algorithm),
      vehicle_routing_algorithm: new FormControl(vehicle_routing_algorithm),
      knapsack_algorithm: new FormControl(knapsack_algorithm),
      vector_search_algorithm: new FormControl(vector_search_algorithm),
      // Num Nonces
      satisfiability_num_nonces: new FormControl(
        data.precommit_manager_config.algo_selection.satisfiability.num_nonces
      ),
      vehicle_routing_num_nonces: new FormControl(
        data.precommit_manager_config.algo_selection.vehicle_routing.num_nonces
      ),
      knapsack_num_nonces: new FormControl(
        data.precommit_manager_config.algo_selection.knapsack.num_nonces
      ),
      vector_search_num_nonces: new FormControl(
        data.precommit_manager_config.algo_selection.vector_search.num_nonces
      ),
      // Weight
      satisfiability_weight: new FormControl(
        data.precommit_manager_config.algo_selection.satisfiability.weight
      ),
      vehicle_routing_weight: new FormControl(
        data.precommit_manager_config.algo_selection.vehicle_routing.weight
      ),
      knapsack_weight: new FormControl(
        data.precommit_manager_config.algo_selection.knapsack.weight
      ),
      vector_search_weight: new FormControl(
        data.precommit_manager_config.algo_selection.vector_search.weight
      ),

      // Difficulty
      satisfiability_difficulty: new FormControl(
        data.difficulty_sampler_config.difficulty_ranges.satisfiability
      ),
      vehicle_routing_difficulty: new FormControl(
        data.difficulty_sampler_config.difficulty_ranges.vehicle_routing
      ),
      knapsack_difficulty: new FormControl(
        data.difficulty_sampler_config.difficulty_ranges.knapsack
      ),
      vector_search_difficulty: new FormControl(
        data.difficulty_sampler_config.difficulty_ranges.vector_search
      ),
    });

    this.settings.set(data);
  }

  async saveForm() {
    const save_data = this.tigService.config();
    // Batch Sizes
    save_data.precommit_manager_config.algo_selection.satisfiability.num_nonces =
      this.formGroup.value.satisfiability_batch_size;
    save_data.precommit_manager_config.algo_selection.vehicle_routing.num_nonces =
      this.formGroup.value.vehicle_routing_batch_size;
    save_data.precommit_manager_config.algo_selection.knapsack.num_nonces =
      this.formGroup.value.knapsack_batch_size;
    save_data.precommit_manager_config.algo_selection.vector_search.num_nonces =
      this.formGroup.value.vector_search_batch_size;
    // Algos
    save_data.precommit_manager_config.algo_selection.satisfiability.algorithm =
      this.formGroup.value.satisfiability_algorithm.name;
    save_data.precommit_manager_config.algo_selection.vehicle_routing.algorithm =
      this.formGroup.value.vehicle_routing_algorithm.name;
    save_data.precommit_manager_config.algo_selection.knapsack.algorithm =
      this.formGroup.value.knapsack_algorithm.name;
    save_data.precommit_manager_config.algo_selection.vector_search.algorithm =
      this.formGroup.value.vector_search_algorithm.name;
    // Num Nonces
    save_data.precommit_manager_config.algo_selection.satisfiability.num_nonces =
      this.formGroup.value.satisfiability_num_nonces;
    save_data.precommit_manager_config.algo_selection.vehicle_routing.num_nonces =
      this.formGroup.value.vehicle_routing_num_nonces;
    save_data.precommit_manager_config.algo_selection.knapsack.num_nonces =
      this.formGroup.value.knapsack_num_nonces;
    save_data.precommit_manager_config.algo_selection.vector_search.num_nonces =
      this.formGroup.value.vector_search_num_nonces;
    // Weight
    save_data.precommit_manager_config.algo_selection.satisfiability.weight =
      this.formGroup.value.satisfiability_weight;
    save_data.precommit_manager_config.algo_selection.vehicle_routing.weight =
      this.formGroup.value.vehicle_routing_weight;
    save_data.precommit_manager_config.algo_selection.knapsack.weight =
      this.formGroup.value.knapsack_weight;
    save_data.precommit_manager_config.algo_selection.vector_search.weight =
      this.formGroup.value.vector_search_weight;
    // Difficulty
    save_data.difficulty_sampler_config.difficulty_ranges.satisfiability =
      this.formGroup.value.satisfiability_difficulty;
    save_data.difficulty_sampler_config.difficulty_ranges.vehicle_routing =
      this.formGroup.value.vehicle_routing_difficulty;
    save_data.difficulty_sampler_config.difficulty_ranges.knapsack =
      this.formGroup.value.knapsack_difficulty;
    save_data.difficulty_sampler_config.difficulty_ranges.vector_search =
      this.formGroup.value.vector_search_difficulty;

    // Max Pending Benchmarks
    save_data.precommit_manager_config.max_pending_benchmarks =
      this.formGroup.value.max_pending_benchmarks;

    this.tigService.saveConfig(save_data);
  }
}
