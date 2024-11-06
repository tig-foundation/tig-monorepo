import { Component, inject, signal } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { TableModule } from 'primeng/table';
import { EditSettingsDialogComponent } from '../../components/edit-settings-dialog/edit-settings-dialog.component';
import { IconFieldModule } from 'primeng/iconfield';
import { InputIconModule } from 'primeng/inputicon';
import { TagModule } from 'primeng/tag';
import { CheckboxModule } from 'primeng/checkbox';
import { id } from 'ethers';
import { InputTextModule } from 'primeng/inputtext';
import { TabViewModule } from 'primeng/tabview';
import { TimeConverterPipe } from '../../pipes/time-converter.pipe';
import { TigApisService } from '../../services/tig-apis.service';

@Component({
  selector: 'app-slave-manager',
  standalone: true,
  imports: [
    TableModule,
    ProgressSpinnerModule,
    InputTextModule,
    ButtonModule,
    CheckboxModule,
    TabViewModule,
    TimeConverterPipe,
    CardModule,
    EditSettingsDialogComponent,
    IconFieldModule,
    InputIconModule,
    TagModule,
  ],
  templateUrl: './slave-manager.component.html',
  styleUrl: './slave-manager.component.scss',
})
export class SlaveManagerComponent {
  tigService: any = inject(TigApisService);

  //Challenges Panel
  challenges_data: any = signal(null);
  //Mining pool panel
  active_slaves: any = signal(0);
  mining_panel_slaves: any = signal(null);
  // Challenge Tabs
  challenge_tabs_slave_table: any = signal(null);
  challenge_tabs_batches_table: any = signal(null);

  mining_panel_data: any = signal(null);
  slaves_data: any = signal(null);
  batches_data: any = signal(null);
  ngOnInit() {
    this.getMiningPanelData();
    this.getChallengesData();
  }
  getMiningPanelData() {
    // Get challenges data
    const temp_data = [
      {
        name: 'Slave 1',
        id: 1,
        api_key: '1231321',
        last_updated: '1 hour ago',
        reward: 0.0001,
        reputation: 0.5,
      },
    ];
    this.mining_panel_slaves.set(temp_data);
    this.active_slaves.set(temp_data.length);
  }
  getChallengesData() {
    const challenges_temp_data = [
      {
        name: 'Challenge 1',
        pending: 4,
        in_progress: 2,
        average_time: 2,
        last_updated: '1 hour ago',
      },
      {
        name: 'Challenge 2',
        pending: 1,
        in_progress: 3,
        average_time: 4,
        last_updated: '1 hour ago',
      },
    ];
    this.challenges_data.set(challenges_temp_data);
  }
  changeChallengeView(challenge: any) {
    const selected = this.tigService.challenges()[challenge];
    const batches = this.tigService
      .batches()
      ?.filter((b:any) => b.challenge_id === selected.id);
    const temp_slaves_challenge_data = [
      {
        name: 'Slave 1',
        max_concurrent_batches: 5,
        recent_batches: 4,
        in_progress: 3,
        average_time_per_batch: 42,
      }
    ]
    this.challenge_tabs_slave_table.set(temp_slaves_challenge_data);

    const temp_batches_challenge_data = [
      {
        benchmark_id: 'Slave 1',
        algorithm: 5,
        batch_size: 4,
        status: 3,
        slave: 42,
        elapsed_time: 42,
      }
    ]
    this.challenge_tabs_batches_table.set(temp_batches_challenge_data);
  }
}
